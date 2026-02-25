import os, asyncio, nest_asyncio, logging, json, base64, httpx, math
from datetime import datetime
from aiohttp import web
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from openai import OpenAI

nest_asyncio.apply()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
RPC_URL = os.getenv("RPC_URL", "https://api.devnet.solana.com")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# FIX: Persist wallet keypair - load from env or generate once and save
WALLET_SECRET = os.getenv("WALLET_SECRET")  # Base58 encoded private key

TOKENS = {
    "SOL": "So11111111111111111111111111111111111111112",
    "USDC": "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU"  # FIX: Real devnet USDC mint
}
PAIRS = [("SOL", "USDC")]

JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"  # FIX: Updated to v6
JUPITER_SWAP_API = "https://quote-api.jup.ag/v6/swap"

MODEL_LIST = ["meta-llama/llama-3.3-70b-instruct:free", "openrouter/auto"]
or_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

def main_menu_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🚀 Start Agent", callback_data="run"),
         InlineKeyboardButton("🛑 Stop", callback_data="stop")],
        [InlineKeyboardButton("💼 Wallet", callback_data="wallet"),
         InlineKeyboardButton("📜 History", callback_data="history")],
        [InlineKeyboardButton("📊 Status", callback_data="status")]
    ])

def compute_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    deltas = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
    gains = sum(d for d in deltas[-period:] if d > 0) / period
    losses = abs(sum(d for d in deltas[-period:] if d < 0)) / period
    if losses == 0:
        return 100
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def compute_ema(prices, period=20):
    if len(prices) < period:
        return prices[-1] if prices else 0
    k = 2 / (period + 1)
    ema = prices[-period]
    for price in prices[-period+1:]:
        ema = price * k + ema * (1 - k)
    return ema

def compute_volatility(prices, period=20):
    if len(prices) < period + 1:
        return 0.0
    returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(-period-1, -1)]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    return math.sqrt(variance)

class SolanaAgent:
    def __init__(self, chat_id):
        self.chat_id = chat_id
        # FIX: Load existing wallet or create new one
        if WALLET_SECRET:
            try:
                secret_bytes = base64.b64decode(WALLET_SECRET)
                self.keypair = Keypair.from_bytes(secret_bytes)
                logging.info(f"Loaded existing wallet for chat {chat_id}")
            except:
                self.keypair = Keypair()
                logging.warning(f"Invalid WALLET_SECRET, generated new wallet: {self.keypair.pubkey()}")
        else:
            self.keypair = Keypair()
            logging.info(f"Generated new wallet for chat {chat_id}: {self.keypair.pubkey()}")
            # Print the secret so you can save it!
            secret = base64.b64encode(bytes(self.keypair)).decode()
            logging.info(f"SAVE THIS WALLET_SECRET: {secret}")
        
        self.client = AsyncClient(RPC_URL, commitment=Confirmed)
        self.is_running = False
        self.history = []
        self.active_positions = []
        self.price_history = {f"{b}/{q}": [] for b, q in PAIRS}
        self.loop_count = 0
        self.last_status = "Initializing..."

    async def get_balance(self):
        try:
            res = await self.client.get_balance(self.keypair.pubkey())
            return res.value / 1e9
        except Exception as e:
            logging.error(f"Balance fetch error: {e}")
            return 0.0

    async def fetch_order_book(self, base, quote):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # FIX: Use proper amount format for Jupiter v6
                params = {
                    "inputMint": TOKENS[base],
                    "outputMint": TOKENS[quote],
                    "amount": str(int(0.001 * 1e9)),  # 0.001 SOL for quote
                    "slippageBps": "100"
                }
                resp = await client.get(JUPITER_QUOTE_API, params=params)
                data = resp.json()
                
                if data.get("data") and len(data["data"]) > 0:
                    # FIX: Jupiter v6 response format
                    out_amount = int(data["data"][0].get("outAmount", 0))
                    in_amount = int(data["data"][0].get("inAmount", 1))
                    price = out_amount / in_amount
                    return {"pair": f"{base}/{quote}", "price": price, "raw_data": data}
                else:
                    logging.warning(f"No quote data: {data}")
                    return {"pair": f"{base}/{quote}", "price": 0, "raw_data": data}
        except Exception as e:
            logging.error(f"Order book fetch error: {e}")
            return {"pair": f"{base}/{quote}", "price": 0, "error": str(e)}

    async def fetch_market_snapshots(self):
        snapshots = []
        for base, quote in PAIRS:
            pair_str = f"{base}/{quote}"
            snapshot = await self.fetch_order_book(base, quote)
            
            if snapshot["price"] > 0:
                hist = self.price_history[pair_str]
                hist.append(snapshot["price"])
                if len(hist) > 50:
                    hist.pop(0)
                
                snapshot["rsi"] = compute_rsi(hist)
                snapshot["ema20"] = compute_ema(hist)
                snapshot["volatility"] = compute_volatility(hist)
            else:
                snapshot["rsi"] = 50
                snapshot["ema20"] = 0
                snapshot["volatility"] = 0
            
            snapshot["balance"] = await self.get_balance()
            snapshot["in_position"] = len([p for p in self.active_positions if p["pair"] == pair_str]) > 0
            snapshot["price_history_count"] = len(self.price_history[pair_str])
            snapshots.append(snapshot)
        
        return snapshots

    async def generate_strategy(self, snapshot):
        # FIX: More aggressive prompting for testing
        prompt = f"""You are an aggressive crypto trading bot. Analyze this market data and trade immediately:

Market: {snapshot['pair']}
Price: {snapshot['price']:.6f}
RSI: {snapshot['rsi']:.1f} (0=oversold, 100=overbought, 50=neutral)
EMA20: {snapshot['ema20']:.6f}
Volatility: {snapshot['volatility']:.4f}
Balance: {snapshot['balance']:.4f} SOL
In Position: {snapshot['in_position']}

RULES:
- If RSI < 40: BUY (oversold)
- If RSI > 60: SELL (overbought)  
- If price > EMA20 * 1.02: BUY (uptrend)
- If price < EMA20 * 0.98: SELL (downtrend)
- Otherwise: WAIT

Return ONLY this JSON format:
{{"action": "BUY" or "SELL" or "WAIT", "confidence": 0-100, "tp_pct": 2, "sl_pct": 1, "reason": "brief explanation"}}"""

        for model in MODEL_LIST:
            try:
                logging.info(f"Querying model: {model}")
                res = or_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=15,
                    temperature=0.7
                )
                content = res.choices[0].message.content.strip()
                
                # FIX: Clean up JSON response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                strategy = json.loads(content)
                logging.info(f"LLM Strategy: {strategy}")
                return strategy
                
            except Exception as e:
                logging.error(f"Model {model} failed: {e}")
                continue
        
        # FIX: Default strategy if all models fail
        logging.warning("All models failed, using default WAIT strategy")
        return {"action": "WAIT", "confidence": 0, "reason": "Models failed"}

    async def execute_actual_swap(self, action, base, quote, amount):
        try:
            in_mint = TOKENS[base] if action == "BUY" else TOKENS[quote]
            out_mint = TOKENS[quote] if action == "BUY" else TOKENS[base]
            
            logging.info(f"Executing {action} {amount} {base} for {quote}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # FIX: Jupiter v6 swap parameters
                quote_params = {
                    "inputMint": in_mint,
                    "outputMint": out_mint,
                    "amount": str(int(amount * 1e9)),
                    "slippageBps": "100",
                    "userPublicKey": str(self.keypair.pubkey())
                }
                
                # Get quote first
                quote_resp = await client.get(JUPITER_QUOTE_API, params=quote_params)
                quote_data = quote_resp.json()
                
                if not quote_data.get("data"):
                    logging.error(f"Quote failed: {quote_data}")
                    return None
                
                # Get swap transaction
                swap_body = {
                    "quoteResponse": quote_data,
                    "userPublicKey": str(self.keypair.pubkey()),
                    "wrapAndUnwrapSOL": True,
                    "computeUnitPriceMicroLamports": 100000  # Priority fee
                }
                
                swap_resp = await client.post(JUPITER_SWAP_API, json=swap_body)
                swap_data = swap_resp.json()
                
                if not swap_data.get("swapTransaction"):
                    logging.error(f"Swap API error: {swap_data}")
                    return None
                
                # Sign and send
                raw_tx = base64.b64decode(swap_data["swapTransaction"])
                tx = VersionedTransaction.from_bytes(raw_tx)
                
                # FIX: Get fresh blockhash
                recent_bh = await self.client.get_latest_blockhash()
                tx.message.recent_blockhash = recent_bh.value.blockhash
                
                tx.sign([self.keypair])
                sig = await self.client.send_raw_transaction(tx.serialize())
                
                logging.info(f"Transaction sent: {sig.value}")
                return sig.value
                
        except Exception as e:
            logging.error(f"Swap execution error: {e}")
            return None

    async def check_positions(self, snapshot, bot):
        for pos in list(self.active_positions):
            current_price = snapshot["price"]
            
            # FIX: Check both TP and SL
            hit_tp = current_price >= pos["tp"]
            hit_sl = current_price <= pos["sl"]
            
            if hit_tp or hit_sl:
                reason = "TAKE PROFIT" if hit_tp else "STOP LOSS"
                logging.info(f"Closing position: {reason} at {current_price}")
                
                sig = await self.execute_actual_swap(
                    "SELL", 
                    pos["pair"].split("/")[0], 
                    pos["pair"].split("/")[1], 
                    pos["amount"]
                )
                
                if sig:
                    profit_loss = round((current_price - pos["entry_price"]) * pos["amount"], 4)
                    self.history.append(f"{datetime.now().strftime('%H:%M')} | EXIT {reason} | {pos['pair']} | P/L: {profit_loss} | Tx: {sig[:8]}...")
                    
                    await bot.send_message(
                        self.chat_id,
                        f"💹 **Trade Closed ({reason})** {pos['pair']}\n"
                        f"Entry: {pos['entry_price']:.6f}\n"
                        f"Exit: {current_price:.6f}\n"
                        f"Amount: {pos['amount']}\n"
                        f"P/L: {profit_loss:.4f}\n"
                        f"[View on Solscan](https://solscan.io/tx/{sig}?cluster=devnet)",
                        parse_mode="Markdown",
                        reply_markup=main_menu_keyboard()
                    )
                    self.active_positions.remove(pos)

    def get_status(self):
        positions_str = "\n".join([
            f"• {p['pair']}: Entry {p['entry_price']:.6f}, TP {p['tp']:.6f}, SL {p['sl']:.6f}"
            for p in self.active_positions
        ]) if self.active_positions else "No active positions"
        
        return (
            f"🤖 **Agent Status**\n"
            f"Running: {'Yes' if self.is_running else 'No'}\n"
            f"Loops completed: {self.loop_count}\n"
            f"Price data points: {self.price_history['SOL/USDC']}\n"
            f"Active positions: {len(self.active_positions)}\n"
            f"History entries: {len(self.history)}\n\n"
            f"📊 **Positions:**\n{positions_str}"
        )

async def agent_loop(chat_id: int, bot):
    agent = manager.get_agent(chat_id)
    
    await bot.send_message(
        chat_id,
        "🚀 **Agent Started**\n"
        f"Wallet: `{agent.keypair.pubkey()}`\n"
        "Monitoring SOL/USDC on DevNet...\n"
        "First trade may take 5-10 minutes to gather data.",
        parse_mode="Markdown",
        reply_markup=main_menu_keyboard()
    )
    
    while agent.is_running:
        try:
            agent.loop_count += 1
            logging.info(f"=== Loop {agent.loop_count} for chat {chat_id} ===")
            
            snapshots = await agent.fetch_market_snapshots()
            
            for snapshot in snapshots:
                logging.info(
                    f"Price: {snapshot['price']:.6f}, "
                    f"RSI: {snapshot['rsi']:.1f}, "
                    f"Data points: {snapshot['price_history_count']}, "
                    f"Balance: {snapshot['balance']:.4f}"
                )
                
                # Check existing positions first
                await agent.check_positions(snapshot, bot)
                
                # Only open new positions if not already in one
                if not snapshot["in_position"] and snapshot["price"] > 0:
                    strategy = await agent.generate_strategy(snapshot)
                    
                    if strategy:
                        action = strategy.get("action", "WAIT")
                        confidence = strategy.get("confidence", 0)
                        reason = strategy.get("reason", "No reason given")
                        
                        logging.info(f"Decision: {action} (confidence: {confidence}%) - {reason}")
                        
                        # FIX: Lower threshold for testing - trade if confidence > 30
                        if action in ["BUY", "SELL"] and confidence >= 30:
                            tp_pct = strategy.get("tp_pct", 2)
                            sl_pct = strategy.get("sl_pct", 1)
                            
                            # FIX: Minimum trade amount 0.05 SOL regardless of confidence
                            trade_amount = max(0.05, 0.1 * (confidence / 100))
                            
                            if snapshot["balance"] >= trade_amount:
                                sig = await agent.execute_actual_swap(
                                    action,
                                    snapshot["pair"].split("/")[0],
                                    snapshot["pair"].split("/")[1],
                                    trade_amount
                                )
                                
                                if sig:
                                    if action == "BUY":
                                        agent.active_positions.append({
                                            "pair": snapshot["pair"],
                                            "entry_price": snapshot["price"],
                                            "amount": trade_amount,
                                            "tp": snapshot["price"] * (1 + tp_pct / 100),
                                            "sl": snapshot["price"] * (1 - sl_pct / 100),
                                            "time": datetime.now()
                                        })
                                    
                                    agent.history.append(
                                        f"{datetime.now().strftime('%H:%M')} | {action} | "
                                        f"{snapshot['pair']} | Amount: {trade_amount} | Tx: {sig[:8]}..."
                                    )
                                    
                                    await bot.send_message(
                                        chat_id,
                                        f"✅ **{action} EXECUTED**\n"
                                        f"Pair: {snapshot['pair']}\n"
                                        f"Price: {snapshot['price']:.6f}\n"
                                        f"Amount: {trade_amount:.4f} SOL\n"
                                        f"Reason: {reason}\n"
                                        f"[View on Solscan](https://solscan.io/tx/{sig}?cluster=devnet)",
                                        parse_mode="Markdown",
                                        reply_markup=main_menu_keyboard()
                                    )
                            else:
                                logging.warning(f"Insufficient balance: {snapshot['balance']} < {trade_amount}")
                                await bot.send_message(
                                    chat_id,
                                    f"⚠️ Insufficient balance ({snapshot['balance']:.4f} SOL) for trade",
                                    reply_markup=main_menu_keyboard()
                                )
                        else:
                            logging.info(f"No trade: action={action}, confidence={confidence}")
                
                agent.last_status = f"Last check: Price {snapshot['price']:.6f}, RSI {snapshot['rsi']:.1f}"
            
            # FIX: Shorter sleep for more responsive testing (15 seconds)
            await asyncio.sleep(15)
            
        except Exception as e:
            logging.error(f"Error in agent loop: {e}")
            await bot.send_message(chat_id, f"⚠️ Agent error: {str(e)[:200]}")
            await asyncio.sleep(30)  # Wait longer on error

class AgentManager:
    def __init__(self):
        self.users = {}
        self.tasks = {}  # FIX: Track background tasks
    
    def get_agent(self, chat_id):
        if chat_id not in self.users:
            self.users[chat_id] = SolanaAgent(chat_id)
        return self.users[chat_id]
    
    def start_agent(self, chat_id, bot):
        agent = self.get_agent(chat_id)
        if not agent.is_running:
            agent.is_running = True
            # FIX: Store task reference
            task = asyncio.create_task(agent_loop(chat_id, bot))
            self.tasks[chat_id] = task
            # Add done callback to catch errors
            task.add_done_callback(
                lambda t: logging.error(f"Task failed: {t.exception()}") if t.exception() else None
            )
            return True
        return False
    
    def stop_agent(self, chat_id):
        agent = self.get_agent(chat_id)
        i
