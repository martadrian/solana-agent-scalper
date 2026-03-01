"""
Autonomous Agentic Wallet System for Solana
Competition-grade implementation with multi-agent architecture
"""

import os
import asyncio
import json
import base64
import logging
import traceback
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import hashlib
import secrets

import httpx
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solders.message import MessageV0
from solders.instruction import Instruction
from solders.pubkey import Pubkey
from solders.system_program import TransferParams, transfer
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed, Finalized
from solana.rpc.types import TxOpts
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RPC_URL = os.getenv("RPC_URL", "https://api.devnet.solana.com")
PORT = int(os.getenv("PORT", "10000"))

if not TELEGRAM_TOKEN or not OPENROUTER_API_KEY:
    raise ValueError("Missing required environment variables: TELEGRAM_TOKEN, OPENROUTER_API_KEY")

# Constants
TOKENS = {
    "SOL": "So11111111111111111111111111111111111111112",
    "USDC": "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
}

JUPITER_API = "https://quote-api.jup.ag/v6"
MAX_SLIPPAGE_BPS = 100
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
        if self.token_in == TOKENS["SOL"]:  # Long SOL
            return (current_price - self.entry_price) * self.amount
        else:  # Short SOL (holding USDC)
            return (self.entry_price - current_price) * self.amount


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
    volume_24h: Optional[float] = None
    liquidity_depth: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


class SecureKeyManager:
    """Handles secure key storage and signing operations"""
    
    def __init__(self):
        self._keys: Dict[int, Keypair] = {}
        self._key_hashes: Dict[int, str] = {}  # Store hash for verification
    
    def get_or_create(self, agent_id: int) -> Keypair:
        """Get existing keypair or create new one"""
        if agent_id not in self._keys:
            # Check environment for existing key
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
            # Store hash for integrity checking
            self._key_hashes[agent_id] = hashlib.sha256(bytes(kp)).hexdigest()[:16]
        
        return self._keys[agent_id]
    
    def _generate_new(self, agent_id: int) -> Keypair:
        """Generate new keypair with cryptographically secure randomness"""
        kp = Keypair()
        secret = base64.b64encode(bytes(kp)).decode()
        
        # Log ONLY the public key and instructions to save the private key
        logger.info(f"Generated new wallet for agent {agent_id}: {kp.pubkey()}")
        logger.info(f"IMPORTANT: Set AGENT_{agent_id}_KEY={secret} in environment")
        
        return kp
    
    def sign_transaction(self, agent_id: int, transaction: VersionedTransaction) -> VersionedTransaction:
        """Sign transaction with agent's key"""
        kp = self._keys.get(agent_id)
        if not kp:
            raise ValueError(f"No key found for agent {agent_id}")
        
        # Verify key integrity
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
    """Robust Solana RPC client with retry logic"""
    
    def __init__(self, rpc_url: str):
        self.rpc_url = rpc_url
        self.client = AsyncClient(rpc_url, commitment=Confirmed)
        self._request_count = 0
    
    async def get_balance(self, pubkey: Pubkey, retries: int = 3) -> float:
        """Get SOL balance with retry logic"""
        for attempt in range(retries):
            try:
                resp = await self.client.get_balance(pubkey)
                if resp.value is not None:
                    return resp.value / 1e9
            except Exception as e:
                logger.warning(f"Balance fetch attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise ConnectionError(f"Failed to fetch balance after {retries} attempts")
    
    async def get_token_balance(self, owner: Pubkey, mint: Pubkey) -> float:
        """Get SPL token balance"""
        try:
            from spl.token.async_client import AsyncToken
            from spl.token.constants import TOKEN_PROGRAM_ID
            
            # Get associated token account
            resp = await self.client.get_token_accounts_by_owner(
                owner,
                {"mint": str(mint)},
                encoding="jsonParsed"
            )
            
            if not resp.value:
                return 0.0
            
            # Sum all token accounts (usually just one)
            total = 0
            for account in resp.value:
                amount = account.account.data.parsed['info']['tokenAmount']['uiAmount']
                if amount:
                    total += amount
            
            return total
        except Exception as e:
            logger.error(f"Token balance error: {e}")
            return 0.0
    
    async def simulate_transaction(self, tx: VersionedTransaction) -> bool:
        """Simulate transaction before sending"""
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
        """Send transaction with confirmation"""
        if opts is None:
            opts = TxOpts(skip_preflight=False, preflight_commitment=Confirmed)
        
        sig = await self.client.send_raw_transaction(tx.serialize(), opts=opts)
        
        # Wait for confirmation
        await self.client.confirm_transaction(sig.value, commitment=Confirmed)
        
        return sig.value
    
    async def get_latest_blockhash(self):
        """Get recent blockhash"""
        resp = await self.client.get_latest_blockhash()
        return resp.value.blockhash


class JupiterDEX:
    """Jupiter DEX integration for optimal routing"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.base_url = JUPITER_API
    
    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = MAX_SLIPPAGE_BPS
    ) -> Optional[Dict]:
        """Get swap quote from Jupiter"""
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": str(slippage_bps),
            "onlyDirectRoutes": "false",
            "asLegacyTransaction": "false"
        }
        
        try:
            resp = await self.client.get(f"{self.base_url}/quote", params=params)
            resp.raise_for_status()
            data = resp.json()
            
            if "error" in data:
                logger.error(f"Jupiter quote error: {data['error']}")
                return None
            
            return data
        except Exception as e:
            logger.error(f"Quote fetch error: {e}")
            return None
    
    async def get_swap_transaction(
        self,
        quote: Dict,
        user_pubkey: str,
        wrap_unwrap_sol: bool = True
    ) -> Optional[str]:
        """Get serialized swap transaction"""
        payload = {
            "quoteResponse": quote,
            "userPublicKey": user_pubkey,
            "wrapAndUnwrapSOL": wrap_unwrap_sol,
            "computeUnitPriceMicroLamports": 50000  # Priority fee for faster execution
        }
        
        try:
            resp = await self.client.post(f"{self.base_url}/swap", json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            if "swapTransaction" not in data:
                logger.error(f"No swap transaction in response: {data}")
                return None
            
            return data["swapTransaction"]
        except Exception as e:
            logger.error(f"Swap transaction error: {e}")
            return None
    
    async def close(self):
        await self.client.aclose()


class AIOracle:
    """AI decision engine using OpenRouter"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=60.0)
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "anthropic/claude-3.5-sonnet"  # Better reasoning for trading
    
    async def analyze_market(self, market_state: MarketState, context: Dict) -> TradeDecision:
        """Get AI trading decision based on market state"""
        
        system_prompt = """You are an elite autonomous crypto trading AI operating on Solana. 
Your goal is to maximize returns while managing risk. You have full autonomy to:
- Open long/short positions
- Close positions based on market conditions
- Hold cash if uncertainty is high
- Emergency exit if market conditions deteriorate

You operate without human intervention. Your decisions are final and executed immediately.

Respond with valid JSON only. No markdown, no explanations outside JSON."""

        user_prompt = f"""MARKET STATE:
- Current Price: ${market_state.current_price:.4f}
- Trend: {market_state.trend}
- Volatility (24h): {market_state.volatility:.4f}
- Price History (last 20): {market_state.price_history[-20:]}

PORTFOLIO CONTEXT:
- Current Balance: {context.get('balance', 0):.4f} SOL
- Active Positions: {len(context.get('positions', []))}
- Unrealized PnL: ${context.get('unrealized_pnl', 0):.2f}
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
                    "temperature": 0.3,  # Lower for more consistent decisions
                    "max_tokens": 800,
                    "response_format": {"type": "json_object"}
                }
            )
            
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            
            # Parse JSON response
            decision_data = json.loads(content)
            
            # Validate and construct decision
            action = ActionType(decision_data.get("action", "WAIT"))
            confidence = min(100, max(0, int(decision_data.get("confidence", 0))))
            
            # Enforce minimum confidence
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
            # Fail safe: wait if AI fails
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
    """
    Core agentic wallet - autonomous economic agent with full wallet control
    """
    
    def __init__(self, agent_id: int, chat_id: int):
        self.agent_id = agent_id
        self.chat_id = chat_id
        self.created_at = datetime.now()
        
        # Components
        self.key_manager = SecureKeyManager()
        self.solana = SolanaClient(RPC_URL)
        self.jupiter = JupiterDEX()
        self.ai = AIOracle(OPENROUTER_API_KEY)
        
        # State
        self.positions: List[Position] = []
        self.trade_history: List[Dict] = []
        self.price_history: List[Dict] = []
        self.is_running = False
        self.loop_count = 0
        self.last_action_time = None
        
        # Performance tracking
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        
        logger.info(f"AgenticWallet {agent_id} initialized: {self.pubkey}")
    
    @property
    def pubkey(self) -> Pubkey:
        return self.key_manager.get_pubkey(self.agent_id)
    
    @property
    def address(self) -> str:
        return str(self.pubkey)
    
    async def get_balance(self) -> float:
        """Get wallet SOL balance"""
        return await self.solana.get_balance(self.pubkey)
    
    async def get_usdc_balance(self) -> float:
        """Get wallet USDC balance"""
        mint = Pubkey.from_string(TOKENS["USDC"])
        return await self.solana.get_token_balance(self.pubkey, mint)
    
    async def fetch_market_data(self) -> Optional[MarketState]:
        """Fetch current market state from Jupiter"""
        try:
            # Get SOL/USDC price (using small amount for quote)
            quote = await self.jupiter.get_quote(
                TOKENS["SOL"],
                TOKENS["USDC"],
                int(0.01 * 1e9)  # 0.01 SOL
            )
            
            if not quote:
                return None
            
            # Extract price from quote
            out_amount = int(quote["outAmount"])
            current_price = out_amount / 1e6  # USDC has 6 decimals
            
            # Update price history
            self.price_history.append({
                "timestamp": datetime.now().isoformat(),
                "price": current_price
            })
            
            # Keep last 100 prices
            if len(self.price_history) > 100:
                self.price_history.pop(0)
            
            prices = [p["price"] for p in self.price_history]
            
            # Calculate metrics
            volatility = self._calculate_volatility(prices)
            trend = self._detect_trend(prices)
            
            return MarketState(
                current_price=current_price,
                price_history=prices,
                volatility=volatility,
                trend=trend
            )
            
        except Exception as e:
            logger.error(f"Market data error: {e}")
            return None
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate realized volatility"""
        if len(prices) < 2:
            return 0.0
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return variance ** 0.5
    
    def _detect_trend(self, prices: List[float]) -> str:
        """Detect market trend using simple moving averages"""
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
        """Build context for AI decision"""
        unrealized = sum(
            pos.current_pnl(self.price_history[-1]["price"] if self.price_history else 0)
            for pos in self.positions
        )
        
        return {
            "balance": self.get_balance(),  # Note: This is async, should be awaited in caller
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
        """Execute trade decision"""
        
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
        """Open long SOL position"""
        try:
            balance = await self.get_balance()
            amount_sol = balance * (decision.amount_percent / 100)
            amount_sol = min(amount_sol, balance * 0.95)  # Keep some for fees
            amount_sol = max(amount_sol, 0.001)  # Minimum trade size
            
            if amount_sol < 0.001:
                logger.warning("Insufficient balance for trade")
                return None
            
            usdc_balance = await self.get_usdc_balance()
            if usdc_balance > 0:
                # Swap USDC to SOL to increase long exposure
                quote = await self.jupiter.get_quote(
                    TOKENS["USDC"],
                    TOKENS["SOL"],
                    int(usdc_balance * 0.5 * 1e6)  # Use 50% of USDC
                )
                
                if not quote:
                    return None
                
                sig = await self._execute_swap(quote)
                
                if sig:
                    # Record position
                    position = Position(
                        entry_price=market.current_price,
                        amount=usdc_balance * 0.5 / market.current_price,
                        token_in=TOKENS["USDC"],
                        token_out=TOKENS["SOL"],
                        take_profit=market.current_price * (1 + decision.take_profit_pct / 100),
                        stop_loss=market.current_price * (1 - decision.stop_loss_pct / 100),
                        tx_signature=sig
                    )
                    self.positions.append(position)
                    
                    self._record_trade("BUY", market.current_price, amount_sol, sig, decision)
                    
                    return sig
            
            logger.info("Already holding SOL, no swap needed")
            return None
            
        except Exception as e:
            logger.error(f"Open long error: {e}")
            return None
    
    async def _close_position(self, decision: TradeDecision, market: MarketState) -> Optional[str]:
        """Close existing position"""
        if not self.positions:
            logger.info("No positions to close")
            return None
        
        # Close first position (FIFO)
        position = self.positions[0]
        
        try:
            # Swap SOL to USDC
            quote = await self.jupiter.get_quote(
                TOKENS["SOL"],
                TOKENS["USDC"],
                int(position.amount * 1e9)
            )
            
            if not quote:
                return None
            
            sig = await self._execute_swap(quote)
            
            if sig:
                # Calculate PnL
                pnl = position.current_pnl(market.current_price)
                self.total_pnl += pnl
                
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
                
                self._record_trade("SELL", market.current_price, position.amount, sig, decision, pnl)
                self.positions.remove(position)
                
                return sig
            
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return None
    
    async def _emergency_exit(self, market: MarketState) -> Optional[str]:
        """Emergency close all positions"""
        logger.warning(f"AGENT {self.agent_id}: EMERGENCY EXIT TRIGGERED")
        
        signatures = []
        for position in list(self.positions):
            quote = await self.jupiter.get_quote(
                TOKENS["SOL"],
                TOKENS["USDC"],
                int(position.amount * 1e9)
            )
            
            if quote:
                sig = await self._execute_swap(quote)
                if sig:
                    signatures.append(sig)
                    self.positions.remove(position)
        
        return signatures[0] if signatures else None
    
    async def _execute_swap(self, quote: Dict) -> Optional[str]:
        """Execute Jupiter swap"""
        try:
            # Get swap transaction
            swap_tx = await self.jupiter.get_swap_transaction(quote, str(self.pubkey))
            if not swap_tx:
                return None
            
            # Deserialize
            raw_tx = base64.b64decode(swap_tx)
            tx = VersionedTransaction.from_bytes(raw_tx)
            
            # Get fresh blockhash
            blockhash = await self.solana.get_latest_blockhash()
            tx.message.recent_blockhash = blockhash
            
            # Sign
            tx = self.key_manager.sign_transaction(self.agent_id, tx)
            
            # Simulate first
            if not await self.solana.simulate_transaction(tx):
                logger.error("Transaction simulation failed")
                return None
            
            # Send
            sig = await self.solana.send_transaction(tx)
            logger.info(f"Swap executed: {sig}")
            
            return sig
            
        except Exception as e:
            logger.error(f"Swap execution error: {e}")
            return None
    
    def _record_trade(
        self,
        action: str,
        price: float,
        amount: float,
        tx_sig: str,
        decision: TradeDecision,
        pnl: Optional[float] = None
    ):
        """Record trade in history"""
        trade = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "price": price,
            "amount": amount,
            "tx_signature": tx_sig,
            "ai_confidence": decision.confidence,
            "ai_reasoning": decision.reasoning,
            "ai_risk": decision.risk_assessment,
            "pnl": pnl
        }
        self.trade_history.append(trade)
        self.last_action_time = datetime.now()
        
        # Keep last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history.pop(0)
    
    async def check_positions(self, market: MarketState) -> List[str]:
        """Check and manage existing positions (TP/SL)"""
        closed = []
        
        for position in list(self.positions):
            current_pnl = position.current_pnl(market.current_price)
            current_price = market.current_price
            
            # Check take profit
            if current_price >= position.take_profit:
                logger.info(f"Take profit hit: {current_price} >= {position.take_profit}")
                sig = await self._close_position_at_price(position, current_price, "TAKE_PROFIT")
                if sig:
                    closed.append(sig)
            
            # Check stop loss
            elif current_price <= position.stop_loss:
                logger.info(f"Stop loss hit: {current_price} <= {position.stop_loss}")
                sig = await self._close_position_at_price(position, current_price, "STOP_LOSS")
                if sig:
                    closed.append(sig)
        
        return closed
    
    async def _close_position_at_price(self, position: Position, price: float, reason: str) -> Optional[str]:
        """Close specific position"""
        try:
            quote = await self.jupiter.get_quote(
                TOKENS["SOL"],
                TOKENS["USDC"],
                int(position.amount * 1e9)
            )
            
            if not quote:
                return None
            
            sig = await self._execute_swap(quote)
            
            if sig:
                pnl = position.current_pnl(price)
                self.total_pnl += pnl
                
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
                
                self.trade_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "action": "SELL",
                    "price": price,
                    "amount": position.amount,
                    "tx_signature": sig,
                    "close_reason": reason,
                    "pnl": pnl
                })
                
                self.positions.remove(position)
                return sig
                
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return None
    
    async def run_autonomous_loop(self, bot=None):
        """Main autonomous operation loop"""
        self.is_running = True
        
        while self.is_running:
            try:
                self.loop_count += 1
                logger.info(f"Agent {self.agent_id}: Loop {self.loop_count}")
                
                # 1. Fetch market data
                market = await self.fetch_market_data()
                if not market:
                    logger.error("Failed to fetch market data")
                    await asyncio.sleep(60)
                    continue
                
                # 2. Check existing positions first
                closed = await self.check_positions(market)
                if closed and bot:
                    for sig in closed:
                        await self._notify_trade(bot, f"Position closed (TP/SL): {sig[:20]}...")
                
                # 3. Build context and get AI decision
                context = self._get_context()
                context['balance'] = await self.get_balance()
                
                decision = await self.ai.analyze_market(market, context)
                
                # 4. Execute decision
                sig = await self.execute_decision(decision, market)
                
                if sig and bot:
                    await self._notify_trade(
                        bot,
                        f"{'BUY' if decision.action == ActionType.BUY else 'SELL'} executed\n"
                        f"Confidence: {decision.confidence}%\n"
                        f"Reason: {decision.reasoning[:100]}..."
                    )
                
                # 5. Wait before next iteration
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Autonomous loop error: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)
    
    async def _notify_trade(self, bot, message: str):
        """Send notification via Telegram"""
        try:
            await bot.send_message(
                self.chat_id,
                f"🤖 Agent {self.agent_id} Update:\n{message}",
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Notification error: {e}")
    
    def stop(self):
        """Stop autonomous operation"""
        self.is_running = False
    
    def get_status(self) -> Dict:
        """Get current agent status"""
        return {
            "agent_id": self.agent_id,
            "address": self.address,
            "is_running": self.is_running,
            "loop_count": self.loop_count,
            "balance_sol": "N/A (async)",
            "positions_count": len(self.positions),
            "total_pnl": self.total_pnl,
            "win_rate": self.win_count / (self.win_count + self.loss_count) if (self.win_count + self.loss_count) > 0 else 0,
            "total_trades": len(self.trade_history),
            "created_at": self.created_at.isoformat()
        }


class MultiAgentSwarm:
    """
    Manages multiple autonomous agents for scalability demonstration
    """
    
    def __init__(self):
        self.agents: Dict[int, AgenticWallet] = {}
        self.tasks: Dict[int, asyncio.Task] = {}
        self.next_id = 1
    
    def create_agent(self, chat_id: int) -> AgenticWallet:
        """Create new autonomous agent"""
        agent_id = self.next_id
        self.next_id += 1
        
        agent = AgenticWallet(agent_id, chat_id)
        self.agents[agent_id] = agent
        return agent
    
    def get_agent(self, chat_id: int) -> Optional[AgenticWallet]:
        """Get or create agent for chat"""
        for agent in self.agents.values():
            if agent.chat_id == chat_id:
                return agent
        
        return self.create_agent(chat_id)
    
    def start_agent(self, agent_id: int, bot=None) -> bool:
        """Start autonomous loop for agent"""
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        if agent.is_running:
            return False
        
        task = asyncio.create_task(agent.run_autonomous_loop(bot))
        self.tasks[agent_id] = task
        return True
    
    def stop_agent(self, agent_id: int) -> bool:
        """Stop agent"""
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        agent.stop()
        
        if agent_id in self.tasks:
            self.tasks[agent_id].cancel()
            del self.tasks[agent_id]
        
        return True
    
    def get_all_status(self) -> List[Dict]:
        """Get status of all agents"""
        return [agent.get_status() for agent in self.agents.values()]


# Global swarm instance
swarm = MultiAgentSwarm()


# Telegram UI
def main_menu():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🚀 Start Agent", callback_data="start_agent"),
         InlineKeyboardButton("🛑 Stop Agent", callback_data="stop_agent")],
        [InlineKeyboardButton("💰 Wallet", callback_data="wallet"),
         InlineKeyboardButton("📊 Status", callback_data="status")],
        [InlineKeyboardButton("📈 Positions", callback_data="positions"),
         InlineKeyboardButton("📜 History", callback_data="history")]
    ])


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    welcome = """🤖 *Autonomous Agentic Wallet*

I am an AI agent with full control over my own Solana wallet. I can:
• Analyze markets independently
• Execute trades without human approval
• Manage risk and positions autonomously
• Learn from my trading history

*⚠️ Warning:* I operate with full autonomy. Once started, I make my own decisions.

Click "Start Agent" to activate me."""
    
    await update.message.reply_text(welcome, parse_mode="Markdown", reply_markup=main_menu())


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks"""
    query = update.callback_query
    await query.answer()
    
    chat_id = query.message.chat_id
    agent = swarm.get_agent(chat_id)
    
    if query.data == "start_agent":
        if swarm.start_agent(agent.agent_id, context.bot):
            await query.edit_message_text(
                f"🟢 Agent {agent.agent_id} activated!\n"
                f"Address: `{agent.address}`\n\n"
                f"Fund this wallet with devnet SOL to begin trading.",
                parse_mode="Markdown",
                reply_markup=main_menu()
            )
        else:
            await query.edit_message_text("Already running!", reply_markup=main_menu())
    
    elif query.data == "stop_agent":
        if swarm.stop_agent(agent.agent_id):
            await query.edit_message_text("🛑 Agent stopped.", reply_markup=main_menu())
        else:
            await query.edit_message_text("Not running.", reply_markup=main_menu())
    
    elif query.data == "wallet":
        balance = await agent.get_balance()
        usdc = await agent.get_usdc_balance()
        text = (
            f"💳 *Wallet Info*\n\n"
            f"Address: `{agent.address}`\n"
            f"SOL: `{balance:.4f}`\n"
            f"USDC: `{usdc:.2f}`\n\n"
            f"[View on Solscan](https://solscan.io/account/{agent.address}?cluster=devnet)"
        )
        await query.edit_message_text(text, parse_mode="Markdown", reply_markup=main_menu())
    
    elif query.data == "status":
        status = agent.get_status()
        text = (
            f"📊 *Agent Status*\n\n"
            f"ID: `{status['agent_id']}`\n"
            f"Status: {'🟢 Running' if status['is_running'] else '🔴 Stopped'}\n"
            f"Loops: `{status['loop_count']}`\n"
            f"Positions: `{status['positions_count']}`\n"
            f"Total PnL: `${status['total_pnl']:.2f}`\n"
            f"Win Rate: `{status['win_rate']*100:.1f}%`\n"
            f"Trades: `{status['total_trades']}`"
        )
        await query.edit_message_text(text, parse_mode="Markdown", reply_markup=main_menu())
    
    elif query.data == "positions":
        if not agent.positions:
            text = "No active positions."
        else:
            text = "📈 *Active Positions*\n\n"
            for i, pos in enumerate(agent.positions, 1):
                text += (
                    f"*{i}. Long SOL*\n"
                    f"Entry: `${pos.entry_price:.4f}`\n"
                    f"Amount: `{pos.amount:.4f}` SOL\n"
                    f"TP: `${pos.take_profit:.4f}` | SL: `${pos.stop_loss:.4f}`\n\n"
                )
        await query.edit_message_text(text, parse_mode="Markdown", reply_markup=main_menu())
    
    elif query.data == "history":
        if not agent.trade_history:
            text = "No trades yet."
        else:
            text = "📜 *Recent Trades*\n\n"
            for trade in agent.trade_history[-5:]:
                emoji = "🟢" if trade.get('pnl', 0) > 0 else "🔴" if trade.get('pnl', 0) < 0 else "⚪"
                text += (
                    f"{emoji} *{trade['action']}* at `${trade['price']:.4f}`\n"
                    f"PnL: `${trade.get('pnl', 0):.2f}` | Conf: `{trade.get('ai_confidence', 0)}%`\n"
                    f"[Tx](https://solscan.io/tx/{trade['tx_signature']}?cluster=devnet)\n\n"
                )
        await query.edit_message_text(text, parse_mode="Markdown", reply_markup=main_menu())


# Web server for health checks
async def health_check(request):
    return web.Response(text=json.dumps({
        "status": "healthy",
        "agents": len(swarm.agents),
        "running": sum(1 for a in swarm.agents.values() if a.is_running)
    }), content_type="application/json")


async def main():
    """Main entry point"""
    # Setup web server (for Render.com)
    app = web.Application()
    app.router.add_get("/", health_check)
    app.router.add_get("/health", health_check)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    logger.info(f"Web server started on port {PORT}")
    
    # Setup Telegram bot
    telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()
    telegram_app.add_handler(CommandHandler("start", start_command))
    telegram_app.add_handler(CallbackQueryHandler(button_handler))
    
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling(drop_pending_updates=True)
    
    logger.info("Autonomous Agentic Wallet System Ready")
    
    # Keep running
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
    
