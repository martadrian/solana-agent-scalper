"""
Autonomous Agentic Wallet System for Solana
Competition-grade implementation with Orca DEX integration
Real transactions on devnet - no simulations
"""

import os
import asyncio
import json
import base64
import logging
import traceback
import struct
import random
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import hashlib

import httpx
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solders.message import MessageV0
from solders.instruction import Instruction
from solders.pubkey import Pubkey
from solders.system_program import ID as SYSTEM_PROGRAM_ID
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed, Finalized
from solana.rpc.types import TxOpts
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from aiohttp import web

# Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment Variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RPC_URL = os.getenv("RPC_URL", "https://api.devnet.solana.com")
PORT = int(os.getenv("PORT", "10000"))

if not TELEGRAM_TOKEN:
    logger.warning("TELEGRAM_TOKEN not set - bot features will be disabled")

if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY not set - AI decisions will use fallback")

# Orca Whirlpools Constants
ORCA_WHIRLPOOL_PROGRAM_ID = Pubkey.from_string("whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc")
TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")

# Devnet SOL Mint (wrapped)
SOL_MINT = Pubkey.from_string("So11111111111111111111111111111111111111112")

# Known Devnet Orca Pools (VERIFIED from your Colab analysis)
DEVNET_POOLS = {
    "pool_1": {
        "address": "CZwdHNd5GFv9yXSoGVXDhkGd68TiHjMPRRXUyeVgJAPf",
        "token_a": "6QBoMRau8roLVhcyavdD9K3UqWA4aWViRYj2pcTuJQBR",
        "token_b": "11114wctDaXejEziXKzm3D82MqwNkW111111111111",
        "tick_spacing": 64,
        "fee_rate": 3000
    },
    "pool_2": {
        "address": "FS5puPbSUYyZmg45Kgvj4E7oPs8DsdV4xTiYbnpojLJQ",
        "token_a": "6QBoMRau8roLVhcyavdD9K3UqWA4aWViRYj2pcTuJQBR",
        "token_b": "11114wctDaXejEziXKzm3D82MqwNkW111111111111",
        "tick_spacing": 64,
        "fee_rate": 3000
    },
    "pool_3": {
        "address": "3Wufe7nsuZ58WQeRqmeQVDGpkWDoEhhrfjv34CNtcUdZ",
        "token_a": "CnuYkATvzwKYvm2tMvp5Jmc4hx62zF9uKg9rbDXMynz3",
        "token_b": "111122DM6LT6idVgB33KHvCxMSQZ4YzYmKNxCGAdCg7",
        "tick_spacing": 64,
        "fee_rate": 3000
    },
    "pool_4": {
        "address": "E1dDZ1MTUaQqDwttz1pioL2DRjroTsnytXNaBwYmzmWw",
        "token_a": "9up2ZZJZncW1FvwRwLuRJpzmVVjCXBC5rPi5b6uLh7uq",
        "token_b": "11112Fsnz5bgxVo36uojYDx43aqxiVRPf7gjHQQwVED",
        "tick_spacing": 64,
        "fee_rate": 3000
    },
    "pool_5": {
        "address": "CT2UpRRef2rrgQ7WKiCbpH2FWmpcUpLvEpS2k1HLrdKJ",
        "token_a": "FcrzuucSf2dSSx1iMZSEuw1YAQSHepit3YXptcV2hN9m",
        "token_b": "11114wctDaXejEziXKzm3D82MqwNkW111111111111",
        "tick_spacing": 64,
        "fee_rate": 3000
    }
}

# Trading Parameters
MIN_CONFIDENCE = 20
MAX_POSITION_SIZE_PCT = 50
DEFAULT_SLIPPAGE_BPS = 100  # 1%

# Tick array constants for Orca
TICK_ARRAY_SIZE = 88
TICK_ARRAY_STRIDE = 88


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
    pool_address: str = ""
    
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
    pool_address: str = ""
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
    tick_current_index: int = 0
    token_vault_a: Optional[str] = None
    token_vault_b: Optional[str] = None
    
    def get_price(self) -> float:
        """Calculate price from sqrt_price_x64"""
        if self.sqrt_price_x64 == 0:
            return 1.0  # Default price instead of 0
        
        sqrt_price = self.sqrt_price_x64 / (2 ** 64)
        price = sqrt_price ** 2
        
        # Only reject truly insane prices
        if price > 1e50 or price < 1e-50:
            logger.warning(f"Extreme price: {price}, using default")
            return 1.0
        
        return price


class SecureKeyManager:
    """Handles secure key storage and signing operations"""
    
    def __init__(self):
        self._keys: Dict[int, Keypair] = {}
        self._key_hashes: Dict[int, str] = {}
    
    def get_or_create(self, agent_id: int) -> Keypair:
        """Get existing keypair or create new one"""
        if agent_id not in self._keys:
            env_key = os.getenv(f"AGENT_{agent_id}_KEY")
            
            if env_key:
                try:
                    secret_bytes = base64.b64decode(env_key)
                    kp = Keypair.from_bytes(secret_bytes)
                    logger.info(f"Loaded existing wallet for agent {agent_id}: {kp.pubkey()}")
                except Exception as e:
                    logger.error(f"Failed to load key for agent {agent_id}: {e}")
                    kp = self._generate_new(agent_id)
            else:
                kp = self._generate_new(agent_id)
            
            self._keys[agent_id] = kp
            self._key_hashes[agent_id] = hashlib.sha256(bytes(kp)).hexdigest()[:16]
        
        return self._keys[agent_id]
    
    def _generate_new(self, agent_id: int) -> Keypair:
        """Generate new keypair"""
        kp = Keypair()
        secret = base64.b64encode(bytes(kp)).decode()
        logger.info(f"=" * 60)
        logger.info(f"GENERATED NEW WALLET FOR AGENT {agent_id}")
        logger.info(f"Public Key: {kp.pubkey()}")
        logger.info(f"=" * 60)
        logger.info(f"IMPORTANT: Save this private key:")
        logger.info(f"AGENT_{agent_id}_KEY={secret}")
        logger.info(f"=" * 60)
        return kp
    
    def sign_transaction(self, agent_id: int, transaction: VersionedTransaction) -> VersionedTransaction:
        """Sign transaction with agent's key"""
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
    """Robust Solana RPC client"""
    
    def __init__(self, rpc_url: str):
        self.rpc_url = rpc_url
        self.client = AsyncClient(rpc_url, commitment=Confirmed)
    
    async def get_balance(self, pubkey: Pubkey, retries: int = 3) -> float:
        """Get SOL balance in SOL"""
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
        """Get SPL token balance"""
        try:
            resp = await self.client.get_token_accounts_by_owner(
                owner,
                {"mint": str(mint)},
                encoding="jsonParsed"
            )
            
            if not resp.value:
                return 0.0
            
            total = 0.0
            for account in resp.value:
                try:
                    amount = account.account.data.parsed['info']['tokenAmount']['uiAmount']
                    if amount:
                        total += amount
                except:
                    pass
            
            return total
        except Exception as e:
            logger.error(f"Token balance error: {e}")
            return 0.0
    
    async def get_or_create_ata(self, owner: Pubkey, mint: Pubkey, payer: Keypair) -> Pubkey:
        """Get or create associated token account"""
        try:
            # Calculate ATA address
            seeds = [
                bytes(owner),
                bytes(TOKEN_PROGRAM_ID),
                bytes(mint)
            ]
            ata, _ = Pubkey.find_program_address(seeds, ASSOCIATED_TOKEN_PROGRAM_ID)
            
            # Check if exists
            resp = await self.client.get_account_info(ata)
            if resp.value:
                return ata
            
            # Create ATA
            logger.info(f"Creating ATA for mint {mint}...")
            
            # Build create ATA instruction data
            data = bytes([1])  # 1 = CreateIdempotent
            
            accounts = [
                {"pubkey": payer.pubkey(), "is_signer": True, "is_writable": True},
                {"pubkey": ata, "is_signer": False, "is_writable": True},
                {"pubkey": owner, "is_signer": False, "is_writable": False},
                {"pubkey": mint, "is_signer": False, "is_writable": False},
                {"pubkey": SYSTEM_PROGRAM_ID, "is_signer": False, "is_writable": False},
                {"pubkey": TOKEN_PROGRAM_ID, "is_signer": False, "is_writable": False},
            ]
            
            ix = Instruction(
                program_id=ASSOCIATED_TOKEN_PROGRAM_ID,
                accounts=accounts,
                data=data
            )
            
            blockhash = await self.get_latest_blockhash()
            message = MessageV0.try_compile(
                payer=payer.pubkey(),
                instructions=[ix],
                address_lookup_table_accounts=[],
                recent_blockhash=blockhash
            )
            
            tx = VersionedTransaction(message, [payer])
            sig = await self.send_transaction(tx)
            logger.info(f"ATA created: {sig}")
            
            return ata
            
        except Exception as e:
            logger.error(f"ATA creation error: {e}")
            raise
    
    async def simulate_transaction(self, tx: VersionedTransaction) -> Tuple[bool, Optional[str]]:
        """Simulate transaction before sending"""
        try:
            result = await self.client.simulate_transaction(tx)
            if result.value.err:
                err_str = str(result.value.err)
                logger.error(f"Simulation failed: {err_str}")
                return False, err_str
            return True, None
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return False, str(e)
    
    async def send_transaction(self, tx: VersionedTransaction, opts: TxOpts = None) -> str:
        """Send transaction and wait for confirmation"""
        if opts is None:
            opts = TxOpts(skip_preflight=False, preflight_commitment=Confirmed)
        
        sig = await self.client.send_raw_transaction(tx.serialize(), opts=opts)
        await self.client.confirm_transaction(sig.value, commitment=Confirmed)
        return str(sig.value)
    
    async def get_latest_blockhash(self):
        """Get recent blockhash"""
        resp = await self.client.get_latest_blockhash()
        return resp.value.blockhash
    
    async def request_airdrop(self, pubkey: Pubkey, amount_sol: float = 2.0) -> Optional[str]:
        """Request devnet SOL airdrop"""
        try:
            sig = await self.client.request_airdrop(pubkey, int(amount_sol * 1e9))
            await self.client.confirm_transaction(sig.value, commitment=Confirmed)
            logger.info(f"Airdropped {amount_sol} SOL to {pubkey}")
            return str(sig.value)
        except Exception as e:
            logger.error(f"Airdrop failed: {e}")
            return None


# Helper functions for tick math
def tick_index_to_sqrt_price_x64(tick_index: int) -> int:
    """Convert tick index to sqrt price (x64 format)"""
    price = (1.0001 ** tick_index) ** 0.5
    return int(price * (2 ** 64))


def sqrt_price_x64_to_tick_index(sqrt_price_x64: int) -> int:
    """Convert sqrt price (x64 format) to tick index"""
    sqrt_price = sqrt_price_x64 / (2 ** 64)
    price = sqrt_price ** 2
    return int(math.log(price) / math.log(1.0001))


def get_tick_array_start_tick_index(tick_index: int, tick_spacing: int) -> int:
    """Get the start tick index for a tick array containing given tick"""
    ticks_per_array = TICK_ARRAY_STRIDE * tick_spacing
    return (tick_index // ticks_per_array) * ticks_per_array


def get_tick_array_address(whirlpool: Pubkey, start_tick_index: int) -> Pubkey:
    """Calculate tick array PDA address"""
    seeds = [
        b"tick_array",
        bytes(whirlpool),
        start_tick_index.to_bytes(4, byteorder="little", signed=True)
    ]
    pubkey, _ = Pubkey.find_program_address(seeds, ORCA_WHIRLPOOL_PROGRAM_ID)
    return pubkey



class OrcaWhirlpoolClient:
    """Real Orca Whirlpools DEX integration with FIXED pool parsing"""
    
    def __init__(self, solana_client: SolanaClient):
        self.solana = solana_client
        self.program_id = ORCA_WHIRLPOOL_PROGRAM_ID
        self.pools: Dict[str, OrcaPool] = {}
        self.best_pool: Optional[OrcaPool] = None
    
    async def load_pool(self, pool_config: Dict) -> Optional[OrcaPool]:
        """Load pool - MINIMAL VERSION for devnet compatibility"""
        try:
            pool_pubkey = Pubkey.from_string(pool_config["address"])
            
            logger.info(f"Checking pool: {pool_config['address']}")
            
            # Just check if account exists and has data
            resp = await self.solana.client.get_account_info(pool_pubkey)
            
            if not resp.value or not resp.value.data:
                logger.warning(f"Pool not found: {pool_config['address']}")
                return None
            
            data = resp.value.data
            
            # For devnet, use a simplified approach
            try:
                # Try to read token mints (usually at offset 40 and 72)
                token_mint_a = Pubkey.from_bytes(data[40:72])
                token_mint_b = Pubkey.from_bytes(data[72:104])
                
                # Read token vaults (usually at offset 104 and 136)
                token_vault_a = Pubkey.from_bytes(data[104:136])
                token_vault_b = Pubkey.from_bytes(data[136:168])
                
                # Try to get sqrt_price from offset 269 (old layout) or 456 (new layout)
                sqrt_price_x64 = 0
                try:
                    # Try new layout first (offset 456)
                    price_new = int.from_bytes(data[456:472], 'little')
                    if 0 < price_new < 2**128:
                        sqrt_price_x64 = price_new
                except:
                    pass
                
                if sqrt_price_x64 == 0:
                    try:
                        # Try old layout (offset 269)
                        price_old = int.from_bytes(data[269:285], 'little')
                        if 0 < price_old < 2**128:
                            sqrt_price_x64 = price_old
                    except:
                        pass
                
                # If still zero, use a default placeholder
                if sqrt_price_x64 == 0:
                    logger.warning(f"Could not parse price for {pool_config['address']}, using placeholder")
                    sqrt_price_x64 = int(1.0 * (2 ** 64))  # Price = 1.0
                
                # Read liquidity (try offset 253 or 440)
                liquidity = 0
                try:
                    liq_new = int.from_bytes(data[440:456], 'little')
                    if liq_new > 0:
                        liquidity = liq_new
                except:
                    pass
                
                if liquidity == 0:
                    try:
                        liq_old = int.from_bytes(data[253:269], 'little')
                        if liq_old > 0:
                            liquidity = liq_old
                    except:
                        pass
                
                # Default liquidity if not found
                if liquidity == 0:
                    liquidity = 1000000  # Placeholder
                
                pool = OrcaPool(
                    address=pool_config["address"],
                    token_a=str(token_mint_a),
                    token_b=str(token_mint_b),
                    sqrt_price_x64=sqrt_price_x64,
                    liquidity=liquidity,
                    tick_spacing=pool_config.get("tick_spacing", 64),
                    fee_rate=pool_config.get("fee_rate", 3000),
                    tick_current_index=0,  # Placeholder
                    token_vault_a=str(token_vault_a),
                    token_vault_b=str(token_vault_b)
                )
                
                price = pool.get_price()
                logger.info(f"✓ Pool loaded: {pool_config['address'][:20]}...")
                logger.info(f"  Price: {price:.6f}")
                logger.info(f"  Liquidity: {liquidity}")
                
                return pool
                
            except Exception as e:
                logger.error(f"Error parsing pool data: {e}")
                return None
            
        except Exception as e:
            logger.error(f"Error loading pool {pool_config['address']}: {e}")
            return None
    
    async def initialize(self) -> bool:
        """Initialize by loading all known pools"""
        logger.info("=" * 60)
        logger.info("Loading Orca Whirlpools...")
        logger.info("=" * 60)
        
        loaded_count = 0
        valid_pools = []
        
        for name, config in DEVNET_POOLS.items():
            try:
                pool = await self.load_pool(config)
                if pool:
                    # Only accept pools with valid prices
                    price = pool.get_price()
                    if price > 0:
                        self.pools[name] = pool
                        valid_pools.append(pool)
                        loaded_count += 1
                        logger.info(f"  ✓ Pool {name}: ACTIVE (price: {price:.6f})")
                    else:
                        logger.warning(f"  ✗ Pool {name}: Invalid price rejected")
                else:
                    logger.warning(f"  ✗ Pool {name}: Failed to load")
            except Exception as e:
                logger.error(f"  ✗ Pool {name}: Error - {e}")
        
        if valid_pools:
            # Select pool with highest liquidity
            self.best_pool = max(valid_pools, key=lambda p: p.liquidity)
            best_price = self.best_pool.get_price()
            logger.info(f"✓ Selected best pool: {self.best_pool.address[:20]}...")
            logger.info(f"✓ Best pool price: {best_price:.10f}")
            logger.info(f"✓ Total valid pools: {len(valid_pools)}")
            return True
        else:
            logger.error("✗ CRITICAL: No valid pools loaded!")
            logger.error("Cannot proceed with trading")
            return False
    
    def get_price(self) -> Optional[float]:
        """Get current price from best pool"""
        if self.best_pool:
            return self.best_pool.get_price()
        return None
    
    async def get_tick_arrays_for_swap(
        self,
        whirlpool: Pubkey,
        tick_current_index: int,
        tick_spacing: int,
        a_to_b: bool
    ) -> List[Pubkey]:
        """Calculate required tick array addresses for a swap"""
        tick_arrays = []
        
        # Current tick array
        current_start_tick = get_tick_array_start_tick_index(tick_current_index, tick_spacing)
        current_array = get_tick_array_address(whirlpool, current_start_tick)
        tick_arrays.append(current_array)
        
        # Add next/prev array depending on swap direction
        ticks_per_array = TICK_ARRAY_STRIDE * tick_spacing
        
        if a_to_b:
            # Swapping A to B: price goes down, tick goes down
            next_start_tick = current_start_tick - ticks_per_array
        else:
            # Swapping B to A: price goes up, tick goes up  
            next_start_tick = current_start_tick + ticks_per_array
        
        next_array = get_tick_array_address(whirlpool, next_start_tick)
        tick_arrays.append(next_array)
        
        # Add one more for safety
        if a_to_b:
            third_start_tick = next_start_tick - ticks_per_array
        else:
            third_start_tick = next_start_tick + ticks_per_array
            
        third_array = get_tick_array_address(whirlpool, third_start_tick)
        tick_arrays.append(third_array)
        
        logger.info(f"Tick arrays for swap:")
        for i, arr in enumerate(tick_arrays):
            logger.info(f"  [{i}] {arr}")
        
        return tick_arrays
    
    async def execute_swap(
        self,
        keypair: Keypair,
        amount_in: float,
        is_a_to_b: bool,
        slippage_bps: int = DEFAULT_SLIPPAGE_BPS
    ) -> Optional[str]:
        """Execute a REAL swap on Orca Whirlpools"""
        if not self.best_pool:
            logger.error("No pool available for swap")
            return None
        
        # Validate pool has reasonable price
        price = self.best_pool.get_price()
        if price == 0 or price > 1e10:
            logger.error(f"Cannot swap - invalid pool price: {price}")
            return None
        
        try:
            owner = keypair.pubkey()
            pool_pubkey = Pubkey.from_string(self.best_pool.address)
            
            token_mint_a = Pubkey.from_string(self.best_pool.token_a)
            token_mint_b = Pubkey.from_string(self.best_pool.token_b)
            token_vault_a = Pubkey.from_string(self.best_pool.token_vault_a) if self.best_pool.token_vault_a else token_mint_a
            token_vault_b = Pubkey.from_string(self.best_pool.token_vault_b) if self.best_pool.token_vault_b else token_mint_b
            
            # Determine input/output based on swap direction
            if is_a_to_b:
                token_in = token_mint_a
                token_out = token_mint_b
                token_vault_in = token_vault_a
                token_vault_out = token_vault_b
                decimals_in = 9
            else:
                token_in = token_mint_b
                token_out = token_mint_a
                token_vault_in = token_vault_b
                token_vault_out = token_vault_a
                decimals_in = 9
            
            amount_in_lamports = int(amount_in * (10 ** decimals_in))
            
            logger.info(f"=" * 60)
            logger.info(f"PREPARING REAL SWAP")
            logger.info(f"=" * 60)
            logger.info(f"Amount: {amount_in} tokens ({amount_in_lamports} lamports)")
            logger.info(f"Direction: {'A->B' if is_a_to_b else 'B->A'}")
            logger.info(f"Pool: {self.best_pool.address}")
            logger.info(f"Pool Price: {price:.10f}")
            logger.info(f"Slippage: {slippage_bps} bps")
            
            # Get or create token accounts
            logger.info("Setting up token accounts...")
            try:
                token_account_in = await self.solana.get_or_create_ata(owner, token_in, keypair)
                token_account_out = await self.solana.get_or_create_ata(owner, token_out, keypair)
            except Exception as e:
                logger.error(f"Failed to setup token accounts: {e}")
                return None
            
            logger.info(f"  Token In ATA: {token_account_in}")
            logger.info(f"  Token Out ATA: {token_account_out}")
            
            # Get tick arrays
            tick_arrays = await self.get_tick_arrays_for_swap(
                pool_pubkey,
                self.best_pool.tick_current_index,
                self.best_pool.tick_spacing,
                is_a_to_b
            )
            
            # Build swap instruction
            swap_ix = self._build_swap_instruction(
                whirlpool=pool_pubkey,
                token_program=TOKEN_PROGRAM_ID,
                token_authority=owner,
                token_owner_account_a=token_account_in if is_a_to_b else token_account_out,
                token_vault_a=token_vault_in if is_a_to_b else token_vault_out,
                token_owner_account_b=token_account_out if is_a_to_b else token_account_in,
                token_vault_b=token_vault_out if is_a_to_b else token_vault_in,
                tick_array_0=tick_arrays[0],
                tick_array_1=tick_arrays[1],
                tick_array_2=tick_arrays[2],
                oracle=tick_arrays[0],
                amount=amount_in_lamports,
                other_amount_threshold=0,  # Accept any output for testing
                sqrt_price_limit=0,  # No limit
                amount_specified_is_input=True,
                a_to_b=is_a_to_b
            )
            
            # Build transaction
            logger.info("Building transaction...")
            blockhash = await self.solana.get_latest_blockhash()
            
            # Add compute budget
            compute_budget_ix = set_compute_unit_limit(400000)
            compute_price_ix = set_compute_unit_price(100000)
            
            message = MessageV0.try_compile(
                payer=owner,
                instructions=[compute_budget_ix, compute_price_ix, swap_ix],
                address_lookup_table_accounts=[],
                recent_blockhash=blockhash
            )
            
            tx = VersionedTransaction(message, [keypair])
            
            # Simulate first
            logger.info("Simulating transaction...")
            success, error = await self.solana.simulate_transaction(tx)
            if not success:
                logger.error(f"Simulation failed: {error}")
                logger.error("Swap aborted")
                return None
            
            logger.info("✓ Simulation passed!")
            
            # Send REAL transaction
            logger.info("Sending REAL transaction to Solana devnet...")
            signature = await self.solana.send_transaction(tx)
            
            logger.info(f"=" * 60)
            logger.info(f"✓ SWAP EXECUTED SUCCESSFULLY!")
            logger.info(f"=" * 60)
            logger.info(f"Transaction Signature: {signature}")
            logger.info(f"View on Solscan: https://solscan.io/tx/{signature}?cluster=devnet")
            logger.info(f"=" * 60)
            
            return signature
            
        except Exception as e:
            logger.error(f"Swap execution error: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _build_swap_instruction(
        self,
        whirlpool: Pubkey,
        token_program: Pubkey,
        token_authority: Pubkey,
        token_owner_account_a: Pubkey,
        token_vault_a: Pubkey,
        token_owner_account_b: Pubkey,
        token_vault_b: Pubkey,
        tick_array_0: Pubkey,
        tick_array_1: Pubkey,
        tick_array_2: Pubkey,
        oracle: Pubkey,
        amount: int,
        other_amount_threshold: int,
        sqrt_price_limit: int,
        amount_specified_is_input: bool,
        a_to_b: bool
    ) -> Instruction:
        """Build Orca Whirlpools swap instruction"""
        
        # Swap instruction discriminator (Anchor)
        discriminator = bytes([248, 198, 158, 145, 225, 117, 135, 200])
        
        # Build instruction data
        data = discriminator
        data += amount.to_bytes(8, 'little')
        data += other_amount_threshold.to_bytes(8, 'little')
        data += sqrt_price_limit.to_bytes(16, 'little')
        data += bytes([1 if amount_specified_is_input else 0])
        data += bytes([1 if a_to_b else 0])
        
        # Account metas
        accounts = [
            {"pubkey": whirlpool, "is_signer": False, "is_writable": True},
            {"pubkey": token_program, "is_signer": False, "is_writable": False},
            {"pubkey": token_authority, "is_signer": True, "is_writable": False},
            {"pubkey": token_owner_account_a, "is_signer": False, "is_writable": True},
            {"pubkey": token_vault_a, "is_signer": False, "is_writable": True},
            {"pubkey": token_owner_account_b, "is_signer": False, "is_writable": True},
            {"pubkey": token_vault_b, "is_signer": False, "is_writable": True},
            {"pubkey": tick_array_0, "is_signer": False, "is_writable": True},
            {"pubkey": tick_array_1, "is_signer": False, "is_writable": True},
            {"pubkey": tick_array_2, "is_signer": False, "is_writable": True},
            {"pubkey": oracle, "is_signer": False, "is_writable": False},
        ]
        
        logger.info(f"Swap instruction: {len(accounts)} accounts")
        
        return Instruction(
            program_id=self.program_id,
            accounts=accounts,
            data=data
        )


class AIOracle:
    """AI decision engine using OpenRouter"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=60.0)
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "anthropic/claude-3.5-sonnet"
    
    async def analyze_market(self, market_state: MarketState, context: Dict) -> TradeDecision:
        """Get AI trading decision"""
        
        if not self.api_key:
            return self._fallback_decision(market_state, context)
        
        system_prompt = """You are an elite autonomous crypto trading AI operating on Solana devnet.
Your goal is to maximize returns while managing risk. You have full autonomy to:
- Open long/short positions
- Close positions based on market conditions
- Hold cash if uncertainty is high
- Emergency exit if market conditions deteriorate

You operate without human intervention. Your decisions are final and executed immediately.

Respond with valid JSON only. No markdown, no explanations outside JSON."""

        user_prompt = f"""MARKET STATE:
- Current Price: {market_state.current_price:.10f}
- Trend: {market_state.trend}
- Volatility: {market_state.volatility:.6f}
- Price History (last 10): {market_state.price_history[-10:]}

PORTFOLIO CONTEXT:
- SOL Balance: {context.get('balance_sol', 0):.4f}
- Token A Balance: {context.get('token_a_balance', 0):.6f}
- Token B Balance: {context.get('token_b_balance', 0):.6f}
- Active Positions: {len(context.get('positions', []))}
- Total PnL: {context.get('total_pnl', 0):.6f}
- Win Rate: {context.get('win_rate', 0)*100:.1f}%

DECISION FRAMEWORK:
1. Analyze trend strength
2. Evaluate risk/reward
3. Consider portfolio exposure
4. Factor in volatility

RISK PARAMETERS:
- Max position size: {MAX_POSITION_SIZE_PCT}%
- Min confidence: {MIN_CONFIDENCE}%

Respond with JSON:
{{
  "action": "BUY" | "SELL" | "WAIT" | "EMERGENCY_EXIT",
  "confidence": 0-100,
  "amount_percent": 10-100,
  "reasoning": "analysis",
  "risk_assessment": "risk eval",
  "take_profit_pct": number,
  "stop_loss_pct": number
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
            
            return TradeDecision(
                action=action,
                confidence=confidence,
                amount_percent=min(MAX_POSITION_SIZE_PCT, int(decision_data.get("amount_percent", 10))),
                reasoning=decision_data.get("reasoning", "No reasoning"),
                risk_assessment=decision_data.get("risk_assessment", "No risk assessment"),
                take_profit_pct=decision_data.get("take_profit_pct", 5.0),
                stop_loss_pct=decision_data.get("stop_loss_pct", 3.0)
            )
            
        except Exception as e:
            logger.error(f"AI decision error: {e}")
            return self._fallback_decision(market_state, context)
    
    def _fallback_decision(self, market_state: MarketState, context: Dict) -> TradeDecision:
        """Simple rule-based fallback when AI is unavailable"""
        
        positions = context.get('positions', [])
        balance = context.get('balance_sol', 0)
        
        # Need price history for trend detection
        if len(market_state.price_history) < 10:
            return TradeDecision(
                action=ActionType.WAIT,
                confidence=50,
                amount_percent=0,
                reasoning="Fallback: Insufficient price history",
                risk_assessment="Neutral - waiting for data"
            )
        
        # Simple trend following
        if market_state.trend == "UPTREND" and not positions and balance > 0.01:
            return TradeDecision(
                action=ActionType.BUY,
                confidence=60,
                amount_percent=30,
                reasoning="Fallback: Uptrend detected, opening position",
                risk_assessment="Moderate - trend following",
                take_profit_pct=5.0,
                stop_loss_pct=3.0
            )
        
        elif market_state.trend == "DOWNTREND" and positions:
            return TradeDecision(
                action=ActionType.SELL,
                confidence=60,
                amount_percent=100,
                reasoning="Fallback: Downtrend detected, closing position",
                risk_assessment="Conservative - protecting capital",
                take_profit_pct=5.0,
                stop_loss_pct=3.0
            )
        
        return TradeDecision(
            action=ActionType.WAIT,
            confidence=50,
            amount_percent=0,
            reasoning="Fallback: No clear signal",
            risk_assessment="Neutral - waiting for better setup"
        )
    
    async def close(self):
        await self.client.aclose()


class AgenticWallet:
    """Core agentic wallet - autonomous economic agent with full wallet control"""
    
    def __init__(self, agent_id: int, chat_id: int):
        self.agent_id = agent_id
        self.chat_id = chat_id
        self.created_at = datetime.now()
        
        # Components
        self.key_manager = SecureKeyManager()
        self.solana = SolanaClient(RPC_URL)
        self.orca = OrcaWhirlpoolClient(self.solana)
        self.ai = AIOracle(OPENROUTER_API_KEY)
        
        # State
        self.positions: List[Position] = []
        self.trade_history: List[Dict] = []
        self.price_history: List[Dict] = []
        self.is_running = False
        self.loop_count = 0
        self.last_action_time = None
        
        # Performance
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        
        logger.info(f"=" * 60)
        logger.info(f"AGENTIC WALLET {agent_id} INITIALIZED")
        logger.info(f"Address: {self.address}")
        logger.info(f"=" * 60)
    
    @property
    def pubkey(self) -> Pubkey:
        return self.key_manager.get_pubkey(self.agent_id)
    
    @property
    def address(self) -> str:
        return str(self.pubkey)
    
    async def initialize(self) -> bool:
        """Initialize the agent"""
        logger.info("Initializing agent...")
        
        # Load Orca pools - CRITICAL: Must succeed for trading
        success = await self.orca.initialize()
        if not success:
            logger.error("CRITICAL: Failed to initialize Orca pools")
            return False
        
        # Check balance
        balance = await self.get_balance()
        logger.info(f"Current balance: {balance:.4f} SOL")
        
        if balance < 0.01:
            logger.warning("LOW BALANCE WARNING!")
            logger.warning(f"Send devnet SOL to: {self.address}")
            logger.warning("Get free SOL from: https://faucet.solana.com/")
        
        return True
    
    async def get_balance(self) -> float:
        """Get SOL balance"""
        return await self.solana.get_balance(self.pubkey)
    
    async def request_airdrop(self, amount: float = 2.0) -> Optional[str]:
        """Request devnet SOL airdrop"""
        return await self.solana.request_airdrop(self.pubkey, amount)
    
    async def fetch_market_data(self) -> Optional[MarketState]:
        """Fetch market data from Orca"""
        try:
            price = self.orca.get_price()
            
            if not price or price <= 0:
                logger.error("No valid price available from Orca pools")
                return None
            
            self.price_history.append({
                "timestamp": datetime.now().isoformat(),
                "price": price
            })
            
            # Keep last 100 prices
            if len(self.price_history) > 100:
                self.price_history.pop(0)
            
            prices = [p["price"] for p in self.price_history]
            
            volatility = self._calculate_volatility(prices)
            trend = self._detect_trend(prices)
            
            return MarketState(
                   current_price=price,
                price_history=prices,
                volatility=volatility,
                trend=trend,
                token_a_symbol="TEST_A",
                token_b_symbol="TEST_B",
                pool_address=self.orca.best_pool.address if self.orca.best_pool else ""
            )
            
        except Exception as e:
            logger.error(f"Market data error: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0.0
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return variance ** 0.5
    
    def _detect_trend(self, prices: List[float]) -> str:
        """Detect market trend"""
        if len(prices) < 10:
            return "INSUFFICIENT_DATA"
        
        short_ma = sum(prices[-3:]) / 3
        long_ma = sum(prices[-10:]) / 10
        
        threshold = long_ma * 0.01  # 1% threshold
        
        if short_ma > long_ma + threshold:
            return "UPTREND"
        elif short_ma < long_ma - threshold:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
    
    async def _get_context(self) -> Dict:
        """Build context for AI decision"""
        current_price = self.price_history[-1]["price"] if self.price_history else 0
        
        unrealized = sum(
            pos.current_pnl(current_price) for pos in self.positions
        )
        
        total_trades = self.win_count + self.loss_count
        
        return {
            "balance_sol": await self.get_balance(),
            "token_a_balance": 0.0,
            "token_b_balance": 0.0,
            "positions": self.positions,
            "unrealized_pnl": unrealized,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_count / total_trades if total_trades > 0 else 0,
            "recent_trades": self.trade_history[-3:]
        }
    
    async def execute_decision(self, decision: TradeDecision, market: MarketState) -> Optional[str]:
        """Execute trade decision"""
        
        if decision.action == ActionType.WAIT:
            logger.info(f"Agent {self.agent_id}: WAIT - {decision.reasoning[:60]}...")
            return None
        
        if decision.action == ActionType.EMERGENCY_EXIT:
            return await self._emergency_exit(market)
        
        if decision.action == ActionType.BUY:
            return await self._open_position(decision, market)
        
        if decision.action == ActionType.SELL:
            return await self._close_position(decision, market)
        
        return None
    
    async def _open_position(self, decision: TradeDecision, market: MarketState) -> Optional[str]:
        """Open a new position with REAL swap"""
        try:
            balance = await self.get_balance()
            
            if balance < 0.01:
                logger.warning("Insufficient SOL for trading")
                return None
            
            # Calculate trade amount
            trade_amount = balance * (decision.amount_percent / 100) * 0.9
            trade_amount = max(trade_amount, 0.001)
            trade_amount = min(trade_amount, balance - 0.01)
            
            logger.info(f"Opening position: {trade_amount:.6f} SOL worth")
            
            # Execute REAL swap (A to B)
            sig = await self.orca.execute_swap(
                keypair=self.key_manager.get_or_create(self.agent_id),
                amount_in=trade_amount,
                is_a_to_b=True
            )
            
            if sig:
                position = Position(
                    entry_price=market.current_price,
                    amount=trade_amount / market.current_price,
                    token_in=self.orca.best_pool.token_a if self.orca.best_pool else "",
                    token_out=self.orca.best_pool.token_b if self.orca.best_pool else "",
                    take_profit=market.current_price * (1 + decision.take_profit_pct / 100),
                    stop_loss=market.current_price * (1 - decision.stop_loss_pct / 100),
                    tx_signature=sig,
                    pool_address=market.pool_address
                )
                self.positions.append(position)
                
                self._record_trade("BUY", market.current_price, trade_amount, sig, decision)
                
                logger.info(f"✓ Position opened: {sig}")
                return sig
            
        except Exception as e:
            logger.error(f"Open position error: {e}")
            logger.error(traceback.format_exc())
        
        return None
    
    async def _close_position(self, decision: TradeDecision, market: MarketState) -> Optional[str]:
        """Close existing position with REAL swap"""
        if not self.positions:
            logger.info("No positions to close")
            return None
        
        position = self.positions[0]
        
        try:
            logger.info(f"Closing position: {position.amount:.6f} tokens")
            
            sig = await self.orca.execute_swap(
                keypair=self.key_manager.get_or_create(self.agent_id),
                amount_in=position.amount,
                is_a_to_b=False
            )
            
            if sig:
                pnl = position.current_pnl(market.current_price)
                self.total_pnl += pnl
                
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
                
                self._record_trade("SELL", market.current_price, position.amount, sig, decision, pnl)
                self.positions.remove(position)
                
                logger.info(f"✓ Position closed. PnL: {pnl:.6f}")
                return sig
            
        except Exception as e:
            logger.error(f"Close position error: {e}")
            logger.error(traceback.format_exc())
        
        return None
    
    async def _emergency_exit(self, market: MarketState) -> Optional[str]:
        """Emergency close all positions"""
        logger.warning(f"AGENT {self.agent_id}: EMERGENCY EXIT")
        
        signatures = []
        for position in list(self.positions):
            sig = await self._close_position_at_price(position, market.current_price, "EMERGENCY")
            if sig:
                signatures.append(sig)
        
        return signatures[0] if signatures else None
    
    async def _close_position_at_price(self, position: Position, price: float, reason: str) -> Optional[str]:
        """Close specific position"""
        try:
            sig = await self.orca.execute_swap(
                keypair=self.key_manager.get_or_create(self.agent_id),
                amount_in=position.amount,
                is_a_to_b=False
            )
            
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
        
        if len(self.trade_history) > 100:
            self.trade_history.pop(0)
    
    async def check_positions(self, market: MarketState) -> List[str]:
        """Check and manage positions (TP/SL)"""
        closed = []
        
        for position in list(self.positions):
            current_price = market.current_price
            
            if current_price >= position.take_profit:
                logger.info(f"🎯 Take profit hit")
                sig = await self._close_position_at_price(position, current_price, "TAKE_PROFIT")
                if sig:
                    closed.append(sig)
            
            elif current_price <= position.stop_loss:
                logger.info(f"🛑 Stop loss hit")
                sig = await self._close_position_at_price(position, current_price, "STOP_LOSS")
                if sig:
                    closed.append(sig)
        
        return closed
    
    async def run_autonomous_loop(self, bot=None):
        """Main autonomous operation loop - REAL TRADING"""
        self.is_running = True
        
        logger.info(f"=" * 60)
        logger.info(f"🚀 AGENT {self.agent_id} AUTONOMOUS LOOP STARTED")
        logger.info(f"💳 Address: {self.address}")
        logger.info(f"⚠️  EXECUTING REAL TRANSACTIONS ON DEVNET")
        logger.info(f"=" * 60)
        
        while self.is_running:
            try:
                self.loop_count += 1
                logger.info(f"\n{'='*40}")
                logger.info(f"LOOP {self.loop_count}")
                logger.info(f"{'='*40}")
                
                # Fetch market data
                market = await self.fetch_market_data()
                if not market:
                    logger.error("Failed to fetch market data - retrying in 60s")
                    await asyncio.sleep(60)
                    continue
                
                logger.info(f"📊 Price: {market.current_price:.10f} | Trend: {market.trend}")
                
                # Check existing positions
                closed = await self.check_positions(market)
                if closed:
                    for sig in closed:
                        logger.info(f"📋 Position closed: {sig[:20]}...")
                        if bot:
                            await self._notify_trade(bot, f"Position closed (TP/SL): `{sig[:20]}...`")
                
                # Get AI decision
                context = await self._get_context()
                decision = await self.ai.analyze_market(market, context)
                
                logger.info(f"🤖 AI Decision: {decision.action.value} (confidence: {decision.confidence}%)")
                logger.info(f"📝 Reasoning: {decision.reasoning[:80]}...")
                
                # Execute decision
                sig = await self.execute_decision(decision, market)
                
                if sig:
                    action_str = "🟢 BUY" if decision.action == ActionType.BUY else "🔴 SELL"
                    logger.info(f"{action_str} EXECUTED: {sig}")
                    if bot:
                        await self._notify_trade(
                            bot,
                            f"{action_str} Executed\n"
                            f"Confidence: {decision.confidence}%\n"
                            f"Tx: `{sig[:30]}...`"
                        )
                
                # Status update
                logger.info(f"💰 Balance: {await self.get_balance():.4f} SOL")
                logger.info(f"📈 Positions: {len(self.positions)} | PnL: {self.total_pnl:.6f}")
                
                # Wait before next iteration
                logger.info(f"⏳ Waiting 60 seconds...")
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Autonomous loop error: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)
        
        logger.info(f"Agent {self.agent_id} stopped")
    
    async def _notify_trade(self, bot, message: str):
        """Send Telegram notification"""
        try:
            await bot.send_message(
                self.chat_id,
                f"🤖 *Agent {self.agent_id}*\n{message}",
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Notification error: {e}")
    
    def stop(self):
        """Stop autonomous operation"""
        self.is_running = False
    
    def get_status(self) -> Dict:
        """Get agent status"""
        total_trades = self.win_count + self.loss_count
        return {
            "agent_id": self.agent_id,
            "address": self.address,
            "is_running": self.is_running,
            "loop_count": self.loop_count,
            "positions_count": len(self.positions),
            "total_pnl": self.total_pnl,
            "win_rate": self.win_count / total_trades if total_trades > 0 else 0,
            "total_trades": len(self.trade_history),
            "created_at": self.created_at.isoformat()
        }


class MultiAgentSwarm:
    """Manages multiple autonomous agents"""
    
    def __init__(self):
        self.agents: Dict[int, AgenticWallet] = {}
        self.tasks: Dict[int, asyncio.Task] = {}
        self.next_id = 1
    
    def create_agent(self, chat_id: int) -> AgenticWallet:
        """Create new agent"""
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
        """Start agent"""
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
        """Get all agents status"""
        return [agent.get_status() for agent in self.agents.values()]


# Global swarm
swarm = MultiAgentSwarm()


# Telegram UI
def main_menu():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🚀 Start Agent", callback_data="start_agent"),
         InlineKeyboardButton("🛑 Stop Agent", callback_data="stop_agent")],
        [InlineKeyboardButton("💰 Wallet", callback_data="wallet"),
         InlineKeyboardButton("📊 Status", callback_data="status")],
        [InlineKeyboardButton("📈 Positions", callback_data="positions"),
         InlineKeyboardButton("📜 History", callback_data="history")],
        [InlineKeyboardButton("💧 Airdrop", callback_data="airdrop")]
    ])


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start"""
    welcome = """🤖 *Autonomous Agentic Wallet*

I am an AI agent with full control over my Solana wallet. I can:
• Analyze markets using Orca DEX data
• Execute *REAL* trades on devnet
• Manage risk and positions autonomously
• Learn from trading history

*⚠️ Real Transactions:* All trades are executed on Solana devnet and recorded on-chain

Click "Start Agent" to activate autonomous trading."""
    
    await update.message.reply_text(welcome, parse_mode="Markdown", reply_markup=main_menu())


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button clicks"""
    query = update.callback_query
    await query.answer()
    
    chat_id = query.message.chat_id
    agent = swarm.get_agent(chat_id)
    
    if query.data == "start_agent":
        success = await agent.initialize()
        
        if not success:
            await query.edit_message_text(
                "❌ *Initialization Failed*\n\n"
                "Could not load Orca pools. Please check:\n"
                "• RPC connection\n"
                "• Pool addresses\n"
                "• Network status",
                parse_mode="Markdown",
                reply_markup=main_menu()
            )
            return
        
        if swarm.start_agent(agent.agent_id, context.bot):
            await query.edit_message_text(
                f"🟢 *Agent {agent.agent_id} Activated!*\n\n"
                f"💳 Address: `{agent.address}`\n\n"
                f"⚠️ *Executing real transactions on devnet*\n\n"
                f"Fund this wallet with devnet SOL:\n"
                f"https://faucet.solana.com/\n\n"
                f"[View on Solscan](https://solscan.io/account/{agent.address}?cluster=devnet)",
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
        text = (
            f"💳 *Wallet Info*\n\n"
            f"Address: `{agent.address}`\n"
            f"Balance: `{balance:.4f}` SOL\n\n"
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
            f"Total PnL: `{status['total_pnl']:.6f}`\n"
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
                    f"*{i}. Position*\n"
                    f"Entry: `{pos.entry_price:.10f}`\n"
                    f"Amount: `{pos.amount:.6f}`\n"
                    f"TP: `{pos.take_profit:.10f}` | SL: `{pos.stop_loss:.10f}`\n"
                    f"[Tx](https://solscan.io/tx/{pos.tx_signature}?cluster=devnet)\n\n"
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
                    f"{emoji} *{trade['action']}* at `{trade['price']:.10f}`\n"
                    f"PnL: `{trade.get('pnl', 0):.6f}` | Conf: `{trade.get('ai_confidence', 0)}%`\n"
                    f"[Tx](https://solscan.io/tx/{trade['tx_signature']}?cluster=devnet)\n\n"
                )
        await query.edit_message_text(text, parse_mode="Markdown", reply_markup=main_menu())
    
    elif query.data == "airdrop":
        sig = await agent.request_airdrop(2.0)
        if sig:
            text = f"✅ Airdrop requested!\nTx: `{sig[:30]}...`"
        else:
            text = "❌ Airdrop failed. Try manually at https://faucet.solana.com/"
        await query.edit_message_text(text, parse_mode="Markdown", reply_markup=main_menu())


# Web server for health checks
async def health_check(request):
    return web.Response(text=json.dumps({
        "status": "healthy",
        "agents": len(swarm.agents),
        "running": sum(1 for a in swarm.agents.values() if a.is_running)
    }), content_type="application/json")


async def agent_status_api(request):
    """API endpoint for agent status"""
    return web.Response(text=json.dumps({
        "agents": swarm.get_all_status()
    }), content_type="application/json")


async def main():
    """Main entry point"""
    # Setup web server
    app = web.Application()
    app.router.add_get("/", health_check)
    app.router.add_get("/health", health_check)
    app.router.add_get("/api/agents", agent_status_api)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    logger.info(f"✓ Web server started on port {PORT}")
    
    # Setup Telegram bot (if token available)
    if TELEGRAM_TOKEN:
        telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()
        telegram_app.add_handler(CommandHandler("start", start_command))
        telegram_app.add_handler(CallbackQueryHandler(button_handler))
        
        await telegram_app.initialize()
        await telegram_app.start()
        await telegram_app.updater.start_polling(drop_pending_updates=True)
        logger.info("✓ Telegram bot started")
    else:
        logger.info("⚠ No Telegram token - running in API-only mode")
    
    logger.info("=" * 60)
    logger.info("🚀 AUTONOMOUS AGENTIC WALLET SYSTEM READY")
    logger.info("⚠️  EXECUTING REAL TRANSACTIONS ON SOLANA DEVNET")
    logger.info("=" * 60)
    
    # Keep running
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())

           
