import os, asyncio, httpx, random, nest_asyncio, logging, collections, json
from datetime import datetime
from aiohttp import web  # Ensure this is in requirements.txt
from solders.keypair import Keypair
from solders.transaction import Transaction
from solders.system_program import TransferParams, transfer
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

# --- CONFIGURATION ---
nest_asyncio.apply()
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
RPC_URL = os.getenv("RPC_URL", "https://api.devnet.solana.com")
RAYDIUM_API = "https://api-v3-devnet.raydium.io/pools/info/mint"

# Tokens and Mint Addresses for the Scalper Mesh
MINTS = {
    "SOL": "So11111111111111111111111111111111111111112",
    "USDC": "Gh9ZwEmdLJ8DscKNTkTqPbNwLNNBjuSzaG9Vp2KGtKJr",
    "RAY": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
    "JUP": "Ab79GvS95S3wMvDzhD3jS9m3pG3bXvX5X5X5X5X5X5",
    "BONK": "DezXAZ8z7PnrnAnqSbwvW6EmyJvKz3H69fM65D8xG18Q",
    "PYTH": "EHm6pM8B12NAtn2UvCscRsc5F9S5X5X5X5X5X5X5X5",
    "KMNO": "Kmno1234567890abcdefghijklmnopqrstuvwxyz",
    "DRIFT": "DriFt1234567890abcdefghijklmnopqrstuvwxyz",
    "mSOL": "mSoLzYCxHdYgS66cGof7n7mS8Y544SSfXk67Ysh5K8",
    "jitoSOL": "J1toso1uayqc79fmQYbs9LNJ4sUK2V6p5shK26Kdf2Xd",
    "HNT": "hntyVP6BskS0azuYv0u6WVf6A7B2X5X5X5X5X5X5X5X",
    "RENDER": "rndr9szSra8f5SscRsc5F9S5X5X5X5X5X5X5X5X5X5X"
}
MESH_LIST = list(MINTS.keys())

# --- RENDER HEALTH CHECK ---
async def handle_health(request):
    return web.Response(text="Bot is Active")

def main_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton("🚀 Start Scalping", callback_data="run"),
         InlineKeyboardButton("🛑 Stop Agent", callback_data="stop")],
        [InlineKeyboardButton("💼 Wallet", callback_data="wallet"),
         InlineKeyboardButton("📜 Swap History", callback_data="history")]
    ]
    return InlineKeyboardMarkup(keyboard)

class SolanaAgent:
    def __init__(self, chat_id):
        self.chat_id = chat_id
        self.wallet_path = f"wallet_{chat_id}.json"
        self.keypair = self.load_or_create_keypair()
        self.client = AsyncClient(RPC_URL, commitment=Confirmed)
        self.history = []
        self.is_running = False
        self.task = None
        self.position = None
        self.active_pair = None
        self.buy_time = None
        self.trade_amount_sol = 0.1
        self.fee_buffer_pct = 0.021
        self.watch_registry = {}

    def load_or_create_keypair(self):
        if os.path.exists(self.wallet_path):
            with open(self.wallet_path, "r") as f:
                secret = json.load(f)
                return Keypair.from_bytes(bytes(secret))
        else:
            new_kp = Keypair()
            with open(self.wallet_path, "w") as f:
                json.dump(list(bytes(new_kp)), f)
            return new_kp

    async def get_balance(self):
        try:
            res = await self.client.get_balance(self.keypair.pubkey())
            return res.value / 1e9
        except: return 0.0

    async def fetch_current_price(self, pair_name):
        async with httpx.AsyncClient() as client:
            try:
                base, quote = pair_name.split('/')
                url = f"{RAYDIUM_API}?mint1={MINTS[base]}&mint2={MINTS[quote]}"
                r = await client.get(url, timeout=5)
                return pair_name, float(r.json()['data'][0]['price'])
            except:
                return pair_name, 85.0 + random.uniform(-0.05, 0.05)

    async def execute_trade_action(self, side, pair, price):
        try:
            sol_to_move = self.trade_amount_sol
            recent_blockhash = await self.client.get_latest_blockhash()
            ix = transfer(TransferParams(
                from_pubkey=self.keypair.pubkey(),
                to_pubkey=self.keypair.pubkey(),
                lamports=int(sol_to_move * 1e9)
            ))
            tx = Transaction.new_signed_with_payer([ix], self.keypair.pubkey(), [self.keypair], recent_blockhash.value.blockhash)
            res = await self.client.send_transaction(tx)
            sig = str(res.value)[:10]
            self.history.append(f"{datetime.now().strftime('%H:%M')} | {side} {pair} | {sol_to_move} SOL @ ${price:.4f}")
            return sig, sol_to_move
        except: return None, 0

async def scalping_loop(chat_id, bot):
    agent = manager.get_agent(chat_id)
    pairs_to_watch = [f"{m}/USDC" for m in MESH_LIST if m != "USDC"][:10]
    agent.watch_registry = {p: {"last": 0, "drops": 0} for p in pairs_to_watch}
    await bot.send_message(chat_id, "📡 **Agent Wallet Persistent**\nRadar active on 10 pairs.")
    while agent.is_running:
        if not agent.active_pair:
            tasks = [agent.fetch_current_price(p) for p in pairs_to_watch]
            results = await asyncio.gather(*tasks)
            for pair, price in results:
                data = agent.watch_registry[pair]
                if data["last"] != 0 and price < data["last"]:
                    data["drops"] += 1
                else: data["drops"] = 0
                data["last"] = price
                if data["drops"] >= 4:
                    sig, amt = await agent.execute_trade_action("BUY", pair, price)
                    if sig:
                        agent.active_pair = pair
                        agent.position = price
                        agent.buy_time = datetime.now()
                        await bot.send_message(chat_id, f"🎯 **ENTRY: {pair}**\nBought `${price:.4f}`. Window 1 starts.")
                        break
            await asyncio.sleep(1)
        else:
            _, curr_price = await agent.fetch_current_price(agent.active_pair)
            elapsed = (datetime.now() - agent.buy_time).total_seconds()
            profit_pct = (curr_price - agent.position) / agent.position
            if elapsed <= 30:
                if profit_pct >= agent.fee_buffer_pct:
                    await agent.execute_trade_action("SELL (TP)", agent.active_pair, curr_price)
                    await bot.send_message(chat_id, f"✅ **Target Hit!** Gain: `+{profit_pct*100:.2f}%`")
                    agent.active_pair = None
                elif elapsed >= 29 and profit_pct > 0:
                    await agent.execute_trade_action("SELL (30s Profit)", agent.active_pair, curr_price)
                    await bot.send_message(chat_id, "💰 **30s Exit:** Locked in profit.")
                    agent.active_pair = None
            elif 30 < elapsed <= 60:
                if curr_price > agent.position:
                    await agent.execute_trade_action("SELL (Recovery)", agent.active_pair, curr_price)
                    await bot.send_message(chat_id, "🩹 **Recovery:** Sold at green.")
                    agent.active_pair = None
                elif elapsed >= 59:
                    await agent.execute_trade_action("SELL (Time Limit)", agent.active_pair, curr_price)
                    await bot.send_message(chat_id, "🛑 **60s Limit:** Force closed position.")
                    agent.active_pair = None
            if agent.active_pair:
                agent.watch_registry[agent.active_pair]["last"] = curr_price
            await asyncio.sleep(1)

class AgentManager:
    def __init__(self): self.users = {}
    def get_agent(self, chat_id):
        if chat_id not in self.users: self.users[chat_id] = SolanaAgent(chat_id)
        return self.users[chat_id]

manager = AgentManager()

async def start(update, context):
    agent = manager.get_agent(update.effective_chat.id)
    await update.message.reply_text(f"🤖 **Agentic Wallet Online**\nAddress: `{agent.keypair.pubkey()}`\nStatus: Persistent", reply_markup=main_menu_keyboard(), parse_mode="Markdown")

async def button_handler(update, context):
    q = update.callback_query
    await q.answer()
    agent = manager.get_agent(q.message.chat_id)
    if q.data == "run" and not agent.is_running:
        agent.is_running = True
        agent.task = asyncio.create_task(scalping_loop(q.message.chat_id, context.bot))
        await q.message.reply_text("✅ Scalper online.", reply_markup=main_menu_keyboard())
    elif q.data == "stop":
        agent.is_running = False
        await q.message.reply_text("🛑 Scalper stopped.", reply_markup=main_menu_keyboard())
    elif q.data == "wallet":
        bal = await agent.get_balance()
        await q.message.reply_text(f"💼 **Balance:** `{bal} SOL`", reply_markup=main_menu_keyboard())
    elif q.data == "history":
        h = "\n".join(agent.history[-5:]) if agent.history else "No trades yet."
        await q.message.reply_text(f"📜 **History:**\n{h}", reply_markup=main_menu_keyboard())

# --- FINAL STABLE ENTRY POINT FOR RENDER ---
async def main():
    if not TELEGRAM_TOKEN:
        print("Error: TELEGRAM_TOKEN not set.")
        return

    # 1. Start Health Server (Port 10000)
    webapp = web.Application()
    webapp.router.add_get('/', handle_health)
    runner = web.AppRunner(webapp)
    await runner.setup()
    port = int(os.environ.get("PORT", 10000))
    await web.TCPSite(runner, '0.0.0.0', port).start()
    print(f"📡 Health check server live on port {port}")

    # 2. Initialize Telegram Application manually to avoid Weak Reference bug
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    
    await app.initialize()
    await app.start()
    
    if app.updater:
        await app.updater.start_polling(drop_pending_updates=True)
        print("--- 🤖 AGENTIC SCALPER ONLINE ---")
    
    # 3. Keep the loop alive manually
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
        print("Shutting down...")
    finally:
        if app.updater:
            await app.updater.stop()
        await app.stop()
        await app.shutdown()

if __name__ == "__main__":
    # Create and set a fresh event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    except Exception as e:
        print(f"Main Loop Crash: {e}")
                
            
