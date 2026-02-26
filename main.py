import os, asyncio, nest_asyncio, logging, json, base64, httpx, math, sys, traceback
from datetime import datetime
from aiohttp import web
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from openai import OpenAI

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

nest_asyncio.apply()

# Environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RPC_URL = os.getenv("RPC_URL", "https://api.devnet.solana.com")
PORT = int(os.getenv("PORT", 10000))
WALLET_SECRET = os.getenv("WALLET_SECRET")

if not TELEGRAM_TOKEN:
    logger.error("❌ TELEGRAM_TOKEN not set!")
    sys.exit(1)

logger.info(f"Starting bot... PORT={PORT}")

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

class SolanaAgent:
    def __init__(self, chat_id):
        self.chat_id = chat_id
        self.keypair = None  
        self.client = AsyncClient(RPC_URL, commitment=Confirmed)
        self.is_running = False
        self.history = []
        self.active_positions = []
        self.price_history = {f"{b}/{q}": [] for b, q in PAIRS}
        self.loop_count = 0

    def get_keypair(self):
        if self.keypair is None:
            if WALLET_SECRET:
                try:
                    secret_bytes = base64.b64decode(WALLET_SECRET)
                    self.keypair = Keypair.from_bytes(secret_bytes)
                    logger.info(f"Loaded wallet: {self.keypair.pubkey()}")
                except Exception as e:
                    logger.error(f"Failed to load wallet: {e}")
                    self.keypair = Keypair()
            else:
                self.keypair = Keypair()
                secret = base64.b64encode(bytes(self.keypair)).decode()
                logger.info(f"WALLET_SECRET: {secret}")
        return self.keypair

    async def get_balance(self):
        try:
            kp = self.get_keypair()
            res = await self.client.get_balance(kp.pubkey())
            return res.value / 1e9
        except Exception as e:
            logger.error(f"Balance error: {e}")
            return 0.0

    async def fetch_order_book(self, base, quote):
        """UPDATED: Fixed Jupiter v6 response parsing"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                params = {
                    "inputMint": TOKENS[base],
                    "outputMint": TOKENS[quote],
                    "amount": str(int(0.001 * 1e9)),
                    "slippageBps": "100"
                }
                resp = await client.get(JUPITER_QUOTE_API, params=params)
                quote_data = resp.json()
                
                # Jupiter v6 returns a dict directly, not a list in 'data'
                if quote_data and "outAmount" in quote_data:
                    out_amount = int(quote_data.get("outAmount", 0))
                    in_amount = int(quote_data.get("inAmount", 1))
                    price = out_amount / in_amount if in_amount > 0 else 0
                    return {"pair": f"{base}/{quote}", "price": price}
                else:
                    logger.error(f"Jupiter quote failed: {quote_data}")
                    return {"pair": f"{base}/{quote}", "price": 0}
        except Exception as e:
            logger.error(f"Order book error: {e}")
            return {"pair": f"{base}/{quote}", "price": 0}

    async def fetch_market_snapshots(self):
        snapshots = []
        for base, quote in PAIRS:
            pair_str = f"{base}/{quote}"
            snapshot = await self.fetch_order_book(base, quote)
            if snapshot["price"] > 0:
                hist = self.price_history[pair_str]
                hist.append(snapshot["price"])
                if len(hist) > 50: hist.pop(0)
                snapshot["rsi"] = self.compute_rsi(hist)
                snapshot["ema20"] = self.compute_ema(hist)
                snapshot["volatility"] = self.compute_volatility(hist)
            else:
                snapshot["rsi"], snapshot["ema20"], snapshot["volatility"] = 50, 0, 0
            snapshot["balance"] = await self.get_balance()
            snapshot["in_position"] = len([p for p in self.active_positions if p["pair"] == pair_str]) > 0
            snapshots.append(snapshot)
        return snapshots

    def compute_rsi(self, prices, period=14):
        if len(prices) < period + 1: return 50
        deltas = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
        gains = sum(d for d in deltas[-period:] if d > 0) / period
        losses = abs(sum(d for d in deltas[-period:] if d < 0)) / period
        if losses == 0: return 100
        rs = gains / losses
        return 100 - (100 / (1 + rs))

    def compute_ema(self, prices, period=20):
        if len(prices) < period: return prices[-1] if prices else 0
        k = 2 / (period + 1)
        ema = prices[-period]
        for price in prices[-period+1:]: ema = price * k + ema * (1 - k)
        return ema

    def compute_volatility(self, prices, period=20):
        if len(prices) < period + 1: return 0.0
        returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(-period-1, -1)]
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance)

    async def generate_strategy(self, snapshot):
        prompt = f"""Analyze and return JSON: {{"action": "BUY"/"SELL"/"WAIT", "confidence": 0-100, "tp_pct": 2, "sl_pct": 1}}
Price: {snapshot['price']:.6f}, RSI: {snapshot['rsi']:.1f}, EMA: {snapshot['ema20']:.6f}, Vol: {snapshot['volatility']:.4f}, Bal: {snapshot['balance']:.4f}"""
        for model in MODEL_LIST:
            try:
                res = or_client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}], timeout=15
                )
                content = res.choices[0].message.content.strip()
                if "```json" in content: content = content.split("```json")[1].split("```")[0]
                elif "```" in content: content = content.split("```")[1].split("```")[0]
                return json.loads(content)
            except Exception as e:
                logger.error(f"Model failed: {e}")
                continue
        return {"action": "WAIT", "confidence": 0}

    async def execute_actual_swap(self, action, base, quote, amount):
        try:
            kp = self.get_keypair()
            in_mint = TOKENS[base] if action == "BUY" else TOKENS[quote]
            out_mint = TOKENS[quote] if action == "BUY" else TOKENS[base]
            async with httpx.AsyncClient(timeout=30.0) as client:
                quote_params = {"inputMint": in_mint, "outputMint": out_mint, "amount": str(int(amount * 1e9)), "slippageBps": "100"}
                quote_resp = await client.get(JUPITER_QUOTE_API, params=quote_params)
                quote_data = quote_resp.json()
                # Use directly as v6 returns the dict
                if not quote_data or "outAmount" not in quote_data: return None
                swap_body = {"quoteResponse": quote_data, "userPublicKey": str(kp.pubkey()), "wrapAndUnwrapSOL": True}
                swap_resp = await client.post(JUPITER_SWAP_API, json=swap_body)
                swap_data = swap_resp.json()
                if not swap_data.get("swapTransaction"): return None
                raw_tx = base64.b64decode(swap_data["swapTransaction"])
                tx = VersionedTransaction.from_bytes(raw_tx)
                recent_bh = await self.client.get_latest_blockhash()
                tx.message.recent_blockhash = recent_bh.value.blockhash
                tx.sign([kp])
                sig = await self.client.send_raw_transaction(tx.serialize())
                return sig.value
        except Exception as e:
            logger.error(f"Swap error: {e}")
            return None

    async def check_positions(self, snapshot, bot):
        for pos in list(self.active_positions):
            current_price = snapshot["price"]
            if current_price >= pos["tp"] or current_price <= pos["sl"]:
                reason = "TP" if current_price >= pos["tp"] else "SL"
                sig = await self.execute_actual_swap("SELL", pos["pair"].split("/")[0], pos["pair"].split("/")[1], pos["amount"])
                if sig:
                    pnl = round((current_price - pos["entry_price"]) * pos["amount"], 4)
                    self.history.append(f"{datetime.now().strftime('%H:%M')} | EXIT {reason} | {pos['pair']} | P/L: {pnl}")
                    await bot.send_message(self.chat_id, f"💹 Closed ({reason}) {pos['pair']} P/L: {pnl}\n[Solscan](https://solscan.io/tx/{sig}?cluster=devnet)", parse_mode="Markdown", reply_markup=main_menu_keyboard())
                    self.active_positions.remove(pos)

async def agent_loop(chat_id, bot):
    agent = manager.get_agent(chat_id)
    await bot.send_message(chat_id, "🚀 Agent started...", reply_markup=main_menu_keyboard())
    while agent.is_running:
        try:
            agent.loop_count += 1
            snapshots = await agent.fetch_market_snapshots()
            for snapshot in snapshots:
                await agent.check_positions(snapshot, bot)
                if not snapshot["in_position"] and snapshot["price"] > 0:
                    strategy = await agent.generate_strategy(snapshot)
                    logger.info(f"Strategy: {strategy}")
                    
                    # LOGIC: Threshold set to 1% confidence for Degen execution
                    if strategy and strategy.get("action") == "BUY" and strategy.get("confidence", 0) >= 1:
                        amount = max(0.05, 0.1 * (strategy["confidence"] / 100))
                        if snapshot["balance"] >= amount:
                            sig = await agent.execute_actual_swap("BUY", "SOL", "USDC", amount)
                            if sig:
                                agent.active_positions.append({"pair": "SOL/USDC", "entry_price": snapshot["price"], "amount": amount, "tp": snapshot["price"] * 1.02, "sl": snapshot["price"] * 0.99})
                                await bot.send_message(chat_id, f"✅ BUY {amount:.4f} SOL\n[Solscan](https://solscan.io/tx/{sig}?cluster=devnet)", parse_mode="Markdown", reply_markup=main_menu_keyboard())
            await asyncio.sleep(15)
        except Exception as e:
            logger.error(f"Loop error: {e}")
            await asyncio.sleep(30)

class AgentManager:
    def __init__(self):
        self.users, self.tasks = {}, {}
    def get_agent(self, chat_id):
        if chat_id not in self.users: self.users[chat_id] = SolanaAgent(chat_id)
        return self.users[chat_id]
    def start_agent(self, chat_id, bot):
        agent = self.get_agent(chat_id)
        if not agent.is_running:
            agent.is_running = True
            self.tasks[chat_id] = asyncio.create_task(agent_loop(chat_id, bot))
            return True
        return False
    def stop_agent(self, chat_id):
        agent = self.get_agent(chat_id)
        if agent.is_running:
            agent.is_running = False
            if chat_id in self.tasks: self.tasks[chat_id].cancel(); del self.tasks[chat_id]
            return True
        return False

manager = AgentManager()

async def debug_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try: await button_handler(update, context)
    except Exception as e: logger.error(traceback.format_exc())

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    chat_id = q.message.chat_id
    agent = manager.get_agent(chat_id)
    if q.data == "run":
        if manager.start_agent(chat_id, context.bot): await q.edit_message_text("✅ Agent starting...", reply_markup=main_menu_keyboard())
    elif q.data == "stop":
        if manager.stop_agent(chat_id): await q.edit_message_text("🛑 Stopped", reply_markup=main_menu_keyboard())
    elif q.data == "wallet":
        kp = agent.get_keypair()
        bal = await agent.get_balance()
        secret = base64.b64encode(bytes(kp)).decode()
        text = f"💼 **Wallet**\n`{kp.pubkey()}`\nBalance: `{bal:.4f}` SOL\n🔑 **Secret**:\n`{secret}`"
        await q.edit_message_text(text, parse_mode="Markdown", reply_markup=main_menu_keyboard())
    elif q.data == "history":
        h = "\n".join(agent.history[-10:]) if agent.history else "No trades"
        await q.edit_message_text(f"📜 History:\n```{h}```", parse_mode="Markdown", reply_markup=main_menu_keyboard())
    elif q.data == "status":
        await q.edit_message_text(f"📊 {'🟢 Running' if agent.is_running else '🔴 Stopped'}\nLoops: {agent.loop_count}\nPositions: {len(agent.active_positions)}", reply_markup=main_menu_keyboard())

async def main():
    app = web.Application()
    app.router.add_get('/', lambda r: web.Response(text="Bot Running"))
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '0.0.0.0', PORT).start()
    telegram_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    telegram_app.add_handler(CommandHandler("start", lambda u,c: u.message.reply_text("🤖 AI Trader", reply_markup=main_menu_keyboard())))
    telegram_app.add_handler(CallbackQueryHandler(debug_button_handler))
    async with telegram_app:
        await telegram_app.initialize(); await telegram_app.start(); await telegram_app.updater.start_polling(drop_pending_updates=True)
        while True: await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
            
                    
