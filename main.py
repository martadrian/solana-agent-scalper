import os, asyncio, nest_asyncio, logging, json, base64, httpx, math, sys
from datetime import datetime
from aiohttp import web
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from openai import OpenAI

# CRITICAL: Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # Ensure logs go to stdout for Render
)

# Apply nest_asyncio
nest_asyncio.apply()

# Environment variables with validation
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RPC_URL = os.getenv("RPC_URL", "https://api.devnet.solana.com")
PORT = int(os.getenv("PORT", 10000))
WALLET_SECRET = os.getenv("WALLET_SECRET")

# Validate required env vars
if not TELEGRAM_TOKEN:
    logging.error("❌ TELEGRAM_TOKEN not set!")
    sys.exit(1)
if not OPENROUTER_API_KEY:
    logging.error("❌ OPENROUTER_API_KEY not set!")
    sys.exit(1)

logging.info(f"Starting with RPC: {RPC_URL}")

TOKENS = {
    "SOL": "So11111111111111111111111111111111111111112",
    "USDC": "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU"
}
PAIRS = [("SOL", "USDC")]

JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"
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
        
        # Load or generate wallet
        if WALLET_SECRET:
            try:
                from solders.keypair import Keypair
                import base64
                secret_bytes = base64.b64decode(WALLET_SECRET)
                self.keypair = Keypair.from_bytes(secret_bytes)
                logging.info(f"Loaded wallet for chat {chat_id}: {self.keypair.pubkey()}")
            except Exception as e:
                logging.error(f"Failed to load wallet: {e}")
                self.keypair = Keypair()
                logging.info(f"Generated new wallet: {self.keypair.pubkey()}")
        else:
            self.keypair = Keypair()
            logging.info(f"Generated new wallet for chat {chat_id}: {self.keypair.pubkey()}")
            # Print secret to logs
            secret = base64.b64encode(bytes(self.keypair)).decode()
            logging.info(f"SAVE THIS WALLET_SECRET: {secret}")
        
        self.client = AsyncClient(RPC_URL, commitment=Confirmed)
        self.is_running = False
        self.history = []
        self.active_positions = []
        self.price_history = {f"{b}/{q}": [] for b, q in PAIRS}
        self.loop_count = 0

    async def get_balance(self):
        try:
            res = await self.client.get_balance(self.keypair.pubkey())
            return res.value / 1e9
        except Exception as e:
            logging.error(f"Balance error: {e}")
            return 0.0

    async def fetch_order_book(self, base, quote):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                params = {
                    "inputMint": TOKENS[base],
                    "outputMint": TOKENS[quote],
                    "amount": str(int(0.001 * 1e9)),
                    "slippageBps": "100"
                }
                resp = await client.get(JUPITER_QUOTE_API, params=params)
                data = resp.json()
                
                if data.get("data") and len(data["data"]) > 0:
                    out_amount = int(data["data"][0].get("outAmount", 0))
                    in_amount = int(data["data"][0].get("inAmount", 1))
                    price = out_amount / in_amount if in_amount > 0 else 0
                    return {"pair": f"{base}/{quote}", "price": price}
                return {"pair": f"{base}/{quote}", "price": 0}
        except Exception as e:
            logging.error(f"Order book error: {e}")
            return {"pair": f"{base}/{quote}", "price": 0}

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
            snapshots.append(snapshot)
        
        return snapshots

    async def generate_strategy(self, snapshot):
        prompt = f"""Analyze market data and return JSON with action (BUY/SELL/WAIT), confidence (0-100), tp_pct, sl_pct.

Market: {snapshot['pair']}
Price: {snapshot['price']:.6f}
RSI: {snapshot['rsi']:.1f}
EMA20: {snapshot['ema20']:.6f}
Volatility: {snapshot['volatility']:.4f}
Balance: {snapshot['balance']:.4f} SOL

Return: {{"action": "BUY", "confidence": 80, "tp_pct": 2, "sl_pct": 1}}"""

        for model in MODEL_LIST:
            try:
                res = or_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=15
                )
                content = res.choices[0].message.content.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                return json.loads(content)
            except Exception as e:
                logging.error(f"Model {model} failed: {e}")
                continue
        
        return {"action": "WAIT", "confidence": 0}

    async def execute_actual_swap(self, action, base, quote, amount):
        try:
            in_mint = TOKENS[base] if action == "BUY" else TOKENS[quote]
            out_mint = TOKENS[quote] if action == "BUY" else TOKENS[base]
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get quote
                quote_params = {
                    "inputMint": in_mint,
                    "outputMint": out_mint,
                    "amount": str(int(amount * 1e9)),
                    "slippageBps": "100"
                }
                quote_resp = await client.get(JUPITER_QUOTE_API, params=quote_params)
                quote_data = quote_resp.json()
                
                if not quote_data.get("data"):
                    return None
                
                # Execute swap
                swap_body = {
                    "quoteResponse": quote_data,
                    "userPublicKey": str(self.keypair.pubkey()),
                    "wrapAndUnwrapSOL": True
                }
                
                swap_resp = await client.post(JUPITER_SWAP_API, json=swap_body)
                swap_data = swap_resp.json()
                
                if not swap_data.get("swapTransaction"):
                    return None
                
                raw_tx = base64.b64decode(swap_data["swapTransaction"])
                tx = VersionedTransaction.from_bytes(raw_tx)
                
                recent_bh = await self.client.get_latest_blockhash()
                tx.message.recent_blockhash = recent_bh.value.blockhash
                tx.sign([self.keypair])
                
                sig = await self.client.send_raw_transaction(tx.serialize())
                return sig.value
                
        except Exception as e:
            logging.error(f"Swap error: {e}")
            return None

    async def check_positions(self, snapshot, bot):
        for pos in list(self.active_positions):
            current_price = snapshot["price"]
            hit_tp = current_price >= pos["tp"]
            hit_sl = current_price <= pos["sl"]
            
            if hit_tp or hit_sl:
                reason = "TP" if hit_tp else "SL"
                sig = await self.execute_actual_swap(
                    "SELL", 
                    pos["pair"].split("/")[0], 
                    pos["pair"].split("/")[1], 
                    pos["amount"]
                )
                
                if sig:
                    profit_loss = round((current_price - pos["entry_price"]) * pos["amount"], 4)
                    self.history.append(f"{datetime.now().strftime('%H:%M')} | EXIT {reason} | {pos['pair']} | P/L: {profit_loss}")
                    
                    solscan_url = f"https://solscan.io/tx/{sig}?cluster=devnet"
                    await bot.send_message(
                        self.chat_id,
                        f"💹 **Trade Closed ({reason})**\n"
                        f"Pair: {pos['pair']}\n"
                        f"P/L: {profit_loss:.4f}\n"
                        f"[View on Solscan]({solscan_url})",
                        parse_mode="Markdown",
                        reply_markup=main_menu_keyboard()
                    )
                    self.active_positions.remove(pos)

async def agent_loop(chat_id, bot):
    agent = manager.get_agent(chat_id)
    
    await bot.send_message(
        chat_id,
        "🚀 Agent started. Trading SOL/USDC on DevNet...",
        reply_markup=main_menu_keyboard()
    )
    
    while agent.is_running:
        try:
            agent.loop_count += 1
            logging.info(f"Loop {agent.loop_count}")
            
            snapshots = await agent.fetch_market_snapshots()
            
            for snapshot in snapshots:
                logging.info(f"Price: {snapshot['price']:.6f}, RSI: {snapshot['rsi']:.1f}, Bal: {snapshot['balance']:.4f}")
                
                await agent.check_positions(snapshot, bot)
                
                if not snapshot["in_position"] and snapshot["price"] > 0:
                    strategy = await agent.generate_strategy(snapshot)
                    logging.info(f"Strategy: {strategy}")
                    
                    if strategy and strategy.get("action") in ["BUY", "SELL"]:
                        conf = strategy.get("confidence", 0)
                        if conf >= 30:
                            amount = max(0.05, 0.1 * (conf / 100))
                            
                            if snapshot["balance"] >= amount:
                                sig = await agent.execute_actual_swap(
                                    strategy["action"],
                                    snapshot["pair"].split("/")[0],
                                    snapshot["pair"].split("/")[1],
                                    amount
                                )
                                
                                if sig:
                                    if strategy["action"] == "BUY":
                                        agent.active_positions.append({
                                            "pair": snapshot["pair"],
                                            "entry_price": snapshot["price"],
                                            "amount": amount,
                                            "tp": snapshot["price"] * 1.02,
                                            "sl": snapshot["price"] * 0.99
                                        })
                                    
                                    solscan_url = f"https://solscan.io/tx/{sig}?cluster=devnet"
                                    await bot.send_message(
                                        chat_id,
                                        f"✅ **{strategy['action']} EXECUTED**\n"
                                        f"Amount: {amount:.4f} SOL\n"
                                        f"[View on Solscan]({solscan_url})",
                                        parse_mode="Markdown",
                                        reply_markup=main_menu_keyboard()
                                    )
            
            await asyncio.sleep(15)
            
        except Exception as e:
            logging.error(f"Loop error: {e}")
            await asyncio.sleep(30)

class AgentManager:
    def __init__(self):
        self.users = {}
        self.tasks = {}
    
    def get_agent(self, chat_id):
        if chat_id not in self.users:
            self.users[chat_id] = SolanaAgent(chat_id)
        return self.users[chat_id]
    
    def start_agent(self, chat_id, bot):
        agent = self.get_agent(chat_id)
        if not agent.is_running:
            agent.is_running = True
            task = asyncio.create_task(agent_loop(chat_id, bot))
            self.tasks[chat_id] = task
            return True
        return False
    
    def stop_agent(self, chat_id):
        agent = self.get_agent(chat_id)
        if agent.is_running:
            agent.is_running = False
            if chat_id in self.tasks:
                self.tasks[chat_id].cancel()
                del self.tasks[chat_id]
            return True
        return False

manager = AgentManager()

async def button_handler(update, context):
    q = update.callback_query
    await q.answer()
    
    chat_id = q.message.chat_id
    agent = manager.get_agent(chat_id)
    
    if q.data == "run":
        if manager.start_agent(chat_id, context.bot):
            await q.message.reply_text("✅ Agent starting...")
        else:
            await q.message.reply_text("Already running!")
    
    elif q.data == "stop":
        if manager.stop_agent(chat_id):
            await q.message.reply_text("🛑 Stopped", reply_markup=main_menu_keyboard())
        else:
            await q.message.reply_text("Not running")
    
    elif q.data == "wallet":
        bal = await agent.get_balance()
        addr = str(agent.keypair.pubkey())
        await q.message.reply_text(
            f"💼 Wallet: `{addr}`\nBalance: {bal:.4f} SOL",
            parse_mode="Markdown",
            reply_markup=main_menu_keyboard()
        )
    
    elif q.data == "history":
        h = "\n".join(agent.history[-10:]) if agent.history else "No trades"
        await q.message.reply_text(f"📜 History:\n{h}", reply_markup=main_menu_keyboard())
    
    elif q.data == "status":
        status = "Running" if agent.is_running else "Stopped"
        await q.message.reply_text(
            f"📊 Status: {status}\nLoops: {agent.loop_count}\nPositions: {len(agent.active_positions)}",
            reply_markup=main_menu_keyboard()
        )

async def start_handler(update, context):
    await update.message.reply_text(
        "🤖 Solana Trading Bot\n\nClick buttons below:",
        reply_markup=main_menu_keyboard()
    )

async def health_check(request):
    return web.Response(text=json.dumps({
        "status": "ok",
        "agents": len(manager.users),
        "running": sum(1 for a in manager.users.values() if a.is_running)
    }))

async def main():
    # Start web server FIRST (Render requires this immediately)
    app = web.Application()
    app.router.add_get('/', lambda r: web.Response(text="Bot Running"))
    app.router.add_get('/health', health_check)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    logging.info(f"Web server started on port {PORT}")
    
    # Start Telegram bot
    telegram_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    telegram_app.add_handler(CommandHandler("start", start_handler))
    telegram_app.add_handler(CallbackQueryHandler(button_handler))
    
    logging.info("Starting Telegram bot...")
    
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling(drop_pending_updates=True)
    
    logging.info("Bot is running!")
    
    # Keep alive
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise
