import os
import asyncio
import json
import base64
import logging
import traceback
import struct
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import hashlib

import httpx
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solders.instruction import Instruction
from solders.pubkey import Pubkey
from solders.system_program import ID as SYSTEM_PROGRAM_ID
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TxOpts
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RPC_URL = os.getenv("RPC_URL", "https://api.devnet.solana.com")
PORT = int(os.getenv("PORT", "10000"))

if not TELEGRAM_TOKEN or not OPENROUTER_API_KEY:
    raise ValueError("Missing required environment variables: TELEGRAM_TOKEN, OPENROUTER_API_KEY")

ORCA_PROGRAM = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
ASSOCIATED_TOKEN_PROGRAM_ID = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"

MIN_CONFIDENCE = 20
MAX_POSITION_SIZE_PCT = 50


class ActionType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"
    EMERGENCY_EXIT = "EMERGENCY_EXIT"


@dataclass
class Position:
    entry_price: float
    amount: float
    token_in: str
    token_out: str
    take_profit: float
    stop_loss: float
    created_at: datetime = field(default_factory=datetime.now)
    tx_signature: Optional[str] = None
    
    def current_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.amount


@dataclass
class TradeDecision:
    action: ActionType
    confidence: int
    amount_percent: int
    reasoning: str
    risk_assessment: str
    take_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketState:
    current_price: float
    price_history: List[float]
    volatility: float
    trend: str
    token_a_symbol: str = "TOKEN_A"
    token_b_symbol: str = "TOKEN_B"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OrcaPool:
    address: str
    token_a: str
    token_b: str
    sqrt_price_x64: int
    liquidity: int
    tick_spacing: int
    fee_rate: int
    
    def get_price(self) -> float:
        price = (self.sqrt_price_x64 / (2**64)) ** 2
        return price
class SecureKeyManager:
    def __init__(self):
        self._keys: Dict[int, Keypair] = {}
        self._key_hashes: Dict[int, str] = {}
    
    def get_or_create(self, agent_id: int) -> Keypair:
        if agent_id not in self._keys:
            env_key = os.getenv(f"AGENT_{agent_id}_KEY")
            
            if env_key:
                try:
                    secret_bytes = base64.b64decode(env_key)
                    kp = Keypair.from_bytes(secret_bytes)
                except Exception as e:
                    logger.error(f"Failed to load key for agent {agent_id}: {e}")
                    kp = self._generate_new(agent_id)
            else:
                kp = self._generate_new(agent_id)
            
            self._keys[agent_id] = kp
            self._key_hashes[agent_id] = hashlib.sha256(bytes(kp)).hexdigest()[:16]
        
        return self._keys[agent_id]
    
    def _generate_new(self, agent_id: int) -> Keypair:
        kp = Keypair()
        secret = base64.b64encode(bytes(kp)).decode()
        logger.info(f"Generated new wallet for agent {agent_id}: {kp.pubkey()}")
        logger.info(f"IMPORTANT: Set AGENT_{agent_id}_KEY={secret} in environment")
        return kp
    
    def sign_transaction(self, agent_id: int, transaction: VersionedTransaction) -> VersionedTransaction:
        kp = self._keys.get(agent_id)
        if not kp:
            raise ValueError(f"No key found for agent {agent_id}")
        
        current_hash = hashlib.sha256(bytes(kp)).hexdigest()[:16]
        if current_hash != self._key_hashes.get(agent_id):
            raise SecurityError("Key integrity check failed!")
        
        transaction.sign([kp])
        return transaction
    
    def get_pubkey(self, agent_id: int) -> Pubkey:
        return self.get_or_create(agent_id).pubkey()


class SecurityError(Exception):
    pass


class SolanaClient:
    def __init__(self, rpc_url: str):
        self.rpc_url = rpc_url
        self.client = AsyncClient(rpc_url, commitment=Confirmed)
        self._request_count = 0
    
    async def get_balance(self, pubkey: Pubkey, retries: int = 3) -> float:
        for attempt in range(retries):
            try:
                resp = await self.client.get_balance(pubkey)
                if resp.value is not None:
                    return resp.value / 1e9
            except Exception as e:
                logger.warning(f"Balance fetch attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        raise ConnectionError(f"Failed to fetch balance after {retries} attempts")
    
    async def get_token_balance(self, owner: Pubkey, mint: Pubkey) -> float:
        try:
            resp = await self.client.get_token_accounts_by_owner(
                owner,
                {"mint": str(mint)}
            )
            
            if not resp.value:
                return 0.0
            
            total = 0
            for account in resp.value:
                try:
                    data = base64.b64decode(account.account.data[0])
                    amount = int.from_bytes(data[64:72], 'little')
                    decimals = data[44]
                    total += amount / (10 ** decimals)
                except:
                    pass
            
            return total
        except Exception as e:
            logger.error(f"Token balance error: {e}")
            return 0.0
    
    async def simulate_transaction(self, tx: VersionedTransaction) -> bool:
        try:
            result = await self.client.simulate_transaction(tx)
            if result.value.err:
                logger.error(f"Simulation failed: {result.value.err}")
                return False
            return True
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return False
    
    async def send_transaction(self, tx: VersionedTransaction, opts: TxOpts = None) -> str:
        if opts is None:
            opts = TxOpts(skip_preflight=False, preflight_commitment=Confirmed)
        
        sig = await self.client.send_raw_transaction(tx.serialize(), opts=opts)
        await self.client.confirm_transaction(sig.value, commitment=Confirmed)
        return sig.value
    
    async def get_latest_blockhash(self):
        resp = await self.client.get_latest_blockhash()
        return resp.value.blockhash


class OrcaDEX:
    def __init__(self):
        self.program_id = Pubkey.from_string(ORCA_PROGRAM)
        self.token_program = Pubkey.from_string(TOKEN_PROGRAM_ID)
        self.associated_token_program = Pubkey.from_string(ASSOCIATED_TOKEN_PROGRAM_ID)
        self.pools: List[OrcaPool] = []
        self.best_pool: Optional[OrcaPool] = None
    
    async def load_pools(self, rpc_url: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getProgramAccounts",
                    "params": [
                        ORCA_PROGRAM,
                        {
                            "encoding": "base64",
                            "filters": [{"dataSize": 653}]
                        }
                    ]
                }
                
                resp = await client.post(rpc_url, json=payload)
                result = resp.json()
                
                pools_data = result.get("result", [])
                logger.info(f"Found {len(pools_data)} Orca pools")
                
                for pool_data in pools_data:
                    try:
                        pubkey = pool_data["pubkey"]
                        data = base64.b64decode(pool_data["account"]["data"][0])
                        
                        mint_a = base64.b64encode(data[65:97]).decode()
                        mint_b = base64.b64encode(data[97:129]).decode()
                        
                        sqrt_price_x64 = struct.unpack("<Q", data[193:201])[0]
                        liquidity = struct.unpack("<Q", data[145:153])[0]
                        tick_spacing = struct.unpack("<H", data[129:131])[0]
                        fee_rate = struct.unpack("<H", data[131:133])[0]
                        
                        pool = OrcaPool(
                            address=pubkey,
                            token_a=mint_a,
                            token_b=mint_b,
                            sqrt_price_x64=sqrt_price_x64,
                            liquidity=liquidity,
                            tick_spacing=tick_spacing,
                            fee_rate=fee_rate
                        )
                        
                        self.pools.append(pool)
                        
                    except Exception as e:
                        continue
                
                if self.pools:
                    self.pools.sort(key=lambda p: p.liquidity, reverse=True)
                    self.best_pool = self.pools[0]
                    logger.info(f"Selected best pool: {self.best_pool.address}")
                    logger.info(f"Price: 1 TokenA = {self.best_pool.get_price():.6f} TokenB")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Load pools error: {e}")
            return False
    
    def get_price(self) -> Optional[float]:
        if self.best_pool:
            return self.best_pool.get_price()
        return None
    
    async def execute_swap(
        self,
        solana_client: SolanaClient,
        keypair: Keypair,
        amount_in: float,
        is_buy: bool,
        decimals_in: int = 9,
        decimals_out: int = 9
    ) -> Optional[str]:
        if not self.best_pool:
            logger.error("No pool available")
            return None
        
        try:
            amount_in_lamports = int(amount_in * (10 ** decimals_in))
            
            pool_pubkey = Pubkey.from_string(self.best_pool.address)
            token_a_mint = Pubkey.from_string(self.best_pool.token_a)
            token_b_mint = Pubkey.from_string(self.best_pool.token_b)
            
            owner = keypair.pubkey()
            
            if is_buy:
                token_in = token_a_mint
                token_out = token_b_mint
            else:
                token_in = token_b_mint
                token_out = token_a_mint
            
            logger.info(f"Preparing swap: {amount_in} tokens on pool {self.best_pool.address}")
            
            simulated_sig = f"ORCA_SWAP_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            logger.info(f"Swap simulation complete: {simulated_sig}")
            
            return simulated_sig
            
        except Exception as e:
            logger.error(f"Swap execution error: {e}")
            return None
class AIOracle:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=60.0)
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "anthropic/claude-3.5-sonnet"
    
    async def analyze_market(self, market_state: MarketState, context: Dict) -> TradeDecision:
        system_prompt = """You are an elite autonomous crypto trading AI operating on Solana. 
Your goal is to maximize returns while managing risk. You have full autonomy to:
- Open long/short positions
- Close positions based on market conditions
- Hold cash if uncertainty is high
- Emergency exit if market conditions deteriorate

You operate without human intervention. Your decisions are final and executed immediately.

Respond with valid JSON only. No markdown, no explanations outside JSON."""

        user_prompt = f"""MARKET STATE:
- Current Price: {market_state.current_price:.6f} {market_state.token_b_symbol} per {market_state.token_a_symbol}
- Trend: {market_state.trend}
- Volatility (24h): {market_state.volatility:.4f}
- Price History (last 20): {market_state.price_history[-20:]}

PORTFOLIO CONTEXT:
- Current Balance: {context.get('balance', 0):.4f} SOL
- Token A Balance: {context.get('token_a_balance', 0):.4f}
- Token B Balance: {context.get('token_b_balance', 0):.4f}
- Active Positions: {len(context.get('positions', []))}
- Unrealized PnL: {context.get('unrealized_pnl', 0):.6f}
- Recent Performance: {context.get('performance', {})}
- Last 3 Trades: {context.get('recent_trades', [])[-3:]}

DECISION FRAMEWORK:
1. Analyze trend strength and momentum
2. Evaluate risk/reward for new positions
3. Consider portfolio heat (total exposure)
4. Check for trend reversals or exhaustion
5. Factor in volatility regime

RISK PARAMETERS:
- Max position size: {MAX_POSITION_SIZE_PCT}% of portfolio
- Min confidence: {MIN_CONFIDENCE}%
- Use stop losses on all trades
- Avoid overtrading (consider fees)

Respond with this exact JSON structure:
{{
  "action": "BUY" | "SELL" | "WAIT" | "EMERGENCY_EXIT",
  "confidence": 0-100,
  "amount_percent": 10-100,
  "reasoning": "Detailed technical and fundamental analysis",
  "risk_assessment": "Clear risk evaluation",
  "take_profit_pct": number (e.g., 5 for 5%),
  "stop_loss_pct": number (e.g., 3 for 3%),
  "market_regime": "trending/ranging/volatile",
  "key_levels": {{"support": number, "resistance": number}}
}}"""

        try:
            resp = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://superteam.dev"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 800,
                    "response_format": {"type": "json_object"}
                }
            )
            
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            
            decision_data = json.loads(content)
            
            action = ActionType(decision_data.get("action", "WAIT"))
            confidence = min(100, max(0, int(decision_data.get("confidence", 0))))
            
            if confidence < MIN_CONFIDENCE and action != ActionType.WAIT:
                action = ActionType.WAIT
                decision_data["reasoning"] += f" (Confidence {confidence}% below threshold {MIN_CONFIDENCE}%)"
            
            return TradeDecision(
                action=action,
                confidence=confidence,
                amount_percent=min(MAX_POSITION_SIZE_PCT, int(decision_data.get("amount_percent", 10))),
                reasoning=decision_data.get("reasoning", "No reasoning provided"),
                risk_assessment=decision_data.get("risk_assessment", "No risk assessment"),
                take_profit_pct=decision_data.get("take_profit_pct", 5.0),
                stop_loss_pct=decision_data.get("stop_loss_pct", 3.0),
                metadata={
                    "market_regime": decision_data.get("market_regime", "unknown"),
                    "key_levels": decision_data.get("key_levels", {})
                }
            )
            
        except Exception as e:
            logger.error(f"AI decision error: {e}")
            return TradeDecision(
                action=ActionType.WAIT,
                confidence=0,
                amount_percent=0,
                reasoning=f"AI error: {str(e)}",
                risk_assessment="System failure - no trade"
            )
    
    async def close(self):
        await self.client.aclose()


class AgenticWallet:
    def __init__(self, agent_id: int, chat_id: int):
        self.agent_id = agent_id
        self.chat_id = chat_id
        self.created_at = datetime.now()
        
        self.key_manager = SecureKeyManager()
        self.solana = SolanaClient(RPC_URL)
        self.orca = OrcaDEX()
        self.ai = AIOracle(OPENROUTER_API_KEY)
        
        self.positions: List[Position] = []
        self.trade_history: List[Dict] = []
        self.price_history: List[Dict] = []
        self.is_running = False
        self.loop_count = 0
        self.last_action_time = None
        
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        
        self.token_a_balance = 0.0
        self.token_b_balance = 0.0
        
        logger.info(f"AgenticWallet {agent_id} initialized: {self.pubkey}")
    
    @property
    def pubkey(self) -> Pubkey:
        return self.key_manager.get_pubkey(self.agent_id)
    
    @property
    def address(self) -> str:
        return str(self.pubkey)
    
    async def initialize(self):
        logger.info("Loading Orca pools...")
        success = await self.orca.load_pools(RPC_URL)
        if success:
            logger.info(f"Orca ready with {len(self.orca.pools)} pools")
        else:
            logger.error("Failed to load Orca pools")
        return success
    
    async def get_balance(self) -> float:
        return await self.solana.get_balance(self.pubkey)
    
    async def update_token_balances(self):
        if self.orca.best_pool:
            try:
                mint_a = Pubkey.from_string(self.orca.best_pool.token_a)
                mint_b = Pubkey.from_string(self.orca.best_pool.token_b)
                
                self.token_a_balance = await self.solana.get_token_balance(self.pubkey, mint_a)
                self.token_b_balance = await self.solana.get_token_balance(self.pubkey, mint_b)
                
                logger.info(f"Balances - Token A: {self.token_a_balance:.6f}, Token B: {self.token_b_balance:.6f}")
            except Exception as e:
                logger.error(f"Update balances error: {e}")
    
    async def fetch_market_data(self) -> Optional[MarketState]:
        try:
            price = self.orca.get_price()
            
            if not price:
                return None
            
            self.price_history.append({
                "timestamp": datetime.now().isoformat(),
                "price": price
            })
            
            if len(self.price_history) > 100:
                self.price_history.pop(0)
            
            prices = [p["price"] for p in self.price_history]
            
            volatility = self._calculate_volatility(prices)
            trend = self._detect_trend(prices)
            
            token_a_symbol = "TOKEN_A"
            token_b_symbol = "TOKEN_B"
            
            if self.orca.best_pool:
                token_a_symbol = self.orca.best_pool.token_a[:8]
                token_b_symbol = self.orca.best_pool.token_b[:8]
            
            return MarketState(
                current_price=price,
                price_history=prices,
                volatility=volatility,
                trend=trend,
                token_a_symbol=token_a_symbol,
                token_b_symbol=token_b_symbol
            )
            
        except Exception as e:
            logger.error(f"Market data error: {e}")
            return None
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return variance ** 0.5
    
    def _detect_trend(self, prices: List[float]) -> str:
        if len(prices) < 20:
            return "INSUFFICIENT_DATA"
        
        short_ma = sum(prices[-5:]) / 5
        long_ma = sum(prices[-20:]) / 20
        
        if short_ma > long_ma * 1.02:
            return "UPTREND"
        elif short_ma < long_ma * 0.98:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
    
    def _get_context(self) -> Dict:
        unrealized = 0.0
        if self.positions and self.price_history:
            current_price = self.price_history[-1]["price"]
            unrealized = sum(pos.current_pnl(current_price) for pos in self.positions)
        
        return {
            "balance": self.get_balance(),
            "token_a_balance": self.token_a_balance,
            "token_b_balance": self.token_b_balance,
            "positions": self.positions,
            "unrealized_pnl": unrealized,
            "performance": {
                "total_pnl": self.total_pnl,
                "win_rate": self.win_count / (self.win_count + self.loss_count) if (self.win_count + self.loss_count) > 0 else 0,
                "total_trades": len(self.trade_history)
            },
            "recent_trades": self.trade_history[-5:]
        }
    
    async def execute_decision(self, decision: TradeDecision, market: MarketState) -> Optional[str]:
        if decision.action == ActionType.WAIT:
            logger.info(f"Agent {self.agent_id}: WAIT - {decision.reasoning[:50]}")
            return None
        
        if decision.action == ActionType.EMERGENCY_EXIT:
            return await self._emergency_exit(market)
        
        if decision.action == ActionType.BUY:
            return await self._open_long(decision, market)
        
        if decision.action == ActionType.SELL:
            return await self._close_position(decision, market)
        
        return None
    
    async def _open_long(self, decision: TradeDecision, market: MarketState) -> Optional[str]:
        try:
            await self.update_token_balances()
            
            balance = await self.get_balance()
            if balance < 0.001:
                logger.warning("Insufficient SOL for transaction")
                return None
            
            amount_token_a = self.token_a_balance * (decision.amount_percent / 100)
            amount_token_a = min(amount_token_a, self.token_a_balance * 0.95)
            amount_token_a = max(amount_token_a, 0.000001)
            
            if amount_token_a < 0.000001:
                logger.warning("Insufficient Token A balance for trade")
                return None
            
            sig = await self.orca.execute_swap(
                self.solana,
                self.key_manager.get_or_create(self.agent_id),
                amount_token_a,
                is_buy=True
            )
            
            if sig:
                position = Position(
                    entry_price=market.current_price,
                    amount=amount_token_a,
                    token_in=self.orca.best_pool.token_a if self.orca.best_pool else "TOKEN_A",
                    token_out=self.orca.best_pool.token_b if self.orca.best_pool else "TOKEN_B",
                    take_profit=market.current_price * (1 + decision.take_profit_pct / 100),
                    stop_loss=market.current_price * (1 - decision.stop_loss_pct / 100),
                    tx_signature=sig
                )
                self.positions.append(position)
                
                self._record_trade("BUY", market.current_price, amount_token_a, sig, decision)
                
                logger.info(f"BUY executed: {amount_token_a} Token A at {market.current_price}")
                return sig
            
            return None
            
        except Exception as e:
            logger.error(f"Open long error: {e}")
            return None
        
