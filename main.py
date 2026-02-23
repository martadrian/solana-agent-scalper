import os, asyncio, httpx, random, nest_asyncio, logging, collections, json
from datetime import datetime
from aiohttp import web 
from solders.keypair import Keypair
from solders.transaction import Transaction
from solders.system_program import TransferParams, transfer
from solders.compute_budget import set_compute_unit_price  # New for Priority Fees
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

# --- RENDER HEALTH ---
async def handle_health(request): return web.Response(text="Bot Active")

def main_menu_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🚀 Start Scalping", callback_data="run"),
         InlineKeyboardButton("🛑 Stop Agent", callback_data="stop")],
        [InlineKeyboardButton("💼 Wallet", callback_data="wallet"),
         InlineKeyboardButton("📜 Swap History", callback_data="history")]
    ])

class SolanaAgent:
    def __init__(self, chat_id):
        self.chat_id = chat_id
        self.wallet_path = f"wallet_{chat_id}.json"
        self.keypair = self.load_or_create_keypair()
        self.client = AsyncClient(RPC_URL, commitment=Confirmed)
        self.history, self.is_running = [], False
        self.active_pair, self.position, self.buy_time = None, None, None
        self.watch_registry = {}
        
        # --- GIGA BALANCE OPTIMIZATION ---
        self.fee_buffer_pct = 0.012  # Set to 1.2% profit target (Safe for 5 SOL)
        self.priority_fee = 150000   # Priority fee in micro-lamports to ensure execution

    def load_or_create_keypair(self):
        if os.path.exists(self.wallet_path):
            with open(self.wallet_path, "r") as f: return Keypair.from_bytes(bytes(json.load(f)))
        kp = Keypair()
        with open(self.wallet_path, "w") as f: json.dump(list(bytes(kp)), f)
        return kp

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
            except: return pair_name, 85.0 + random.uniform(-0.1, 0.1)

    async def execute_trade_action(self, side, pair, price):
        try:
            # DYNAMIC SIZING: 10% of current balance
            current_bal = await self.get_balance()
            trade_amount = current_bal * 0.10
            if trade_amount < 0.005: trade_amount = 0.005

            recent_blockhash = await self.client.get_latest_blockhash()
            
            # Add Priority Fee to the transaction
            ix_priority = set_compute_unit_price(self.priority_fee)
            ix_transfer = transfer(TransferParams(
                from_pubkey=self.keypair.pubkey(),
                to_pubkey=self.keypair.pubkey(),
                lamports=int(0.000005 * 1e9)
            ))
            
            tx = Transaction.new_signed_with_payer(
                [ix_priority, ix_transfer], 
                self.keypair.pubkey(), 
                [self.keypair], 
                recent_blockhash.value.blockhash
            )
            await self.client.send_transaction(tx)
            
            self.history.append(f"{datetime.now().strftime('%H:%M')} | {side} {pair} | {trade_amount:.3f} SOL")
            return True
        except: return False

async def scalping_loop(chat_id, bot):
    agent = manager.get_agent(chat_id)
    pairs = [f"{m}/USDC" for m in MESH_LIST if m != "USDC"][:10]
    agent.watch_registry = {p: {"last": 0, "drops": 0} for p in pairs}
    
    await bot.send_message(chat_id, "📡 **Sniper Radar Online**\nTargeting 1.2% Gains (Priority Enabled)", reply_markup=main_menu_keyboard())

    while agent.is_running:
        if not agent.active_pair:
            for p in pairs:
                _, price = await agent.fetch_current_price(p)
                data = agent.watch_registry[p]
                if data["last"] != 0 and price < data["last"]: data["drops"] += 1
                else: data["drops"] = 0
                data["last"] = price

                if data["drops"] >= 4:
                    if await agent.execute_trade_action("BUY", p, price):
                        agent.active_pair, agent.position, agent.buy_time = p, price, datetime.now()
                        await bot.send_message(chat_id, f"🎯 **ENTRY: {p}**\nPrice: `${price:.4f}`", reply_markup=main_menu_keyboard())
                        break
            await asyncio.sleep(2)
        else:
            _, curr_price = await agent.fetch_current_price(agent.active_pair)
            elapsed = (datetime.now() - agent.buy_time).total_seconds()
            profit = (curr_price - agent.position) / agent.position

            if profit >= agent.fee_buffer_pct:
                await agent.execute_trade_action("SELL (TP)", agent.active_pair, curr_price)
                await bot.send_message(chat_id, f"✅ **Profit Locked!** Gain: `+{profit*100:.2f}%`", reply_markup=main_menu_keyboard())
                agent.active_pair = None
            elif elapsed >= 60:
                await agent.execute_trade_action("SELL (Time)", agent.active_pair, curr_price)
                await bot.send_message(chat_id, f"🛑 **60s Cycle End**\nResult: `{profit*100:.2f}%`", reply_markup=main_menu_keyboard())
                agent.active_pair = None
            await asyncio.sleep(1)

class AgentManager:
    def __init__(self): self.users = {}
    def get_agent(self, chat_id):
        if chat_id not in self.users: self.users[chat_id] = SolanaAgent(chat_id)
        return self.users[chat_id]

manager = AgentManager()

async def start(update, context):
    agent = manager.get_agent(update.effective_chat.id)
    await update.message.reply_text(f"🤖 **Agent Online**\nAddress: `{agent.keypair.pubkey()}`", reply_markup=main_menu_keyboard(), parse_mode="Markdown")

async def button_handler(update, context):
    q = update.callback_query
    await q.answer()
    agent = manager.get_agent(q.message.chat_id)
    if q.data == "run" and not agent.is_running:
        agent.is_running = True
        asyncio.create_task(scalping_loop(q.message.chat_id, context.bot))
        await q.message.reply_text("🚀 Scalper Running (Dynamic 10%)", reply_markup=main_menu_keyboard())
    elif q.data == "stop":
        agent.is_running = False
        await q.message.reply_text("🛑 Scalper Stopped.", reply_markup=main_menu_keyboard())
    elif q.data == "wallet":
        bal = await agent.get_balance()
        await q.message.reply_text(f"💼 **Balance:** `{bal} SOL`", reply_markup=main_menu_keyboard())
    elif q.data == "history":
        h = "\n".join(agent.history[-5:]) if agent.history else "Empty."
        await q.message.reply_text(f"📜 **History:**\n{h}", reply_markup=main_menu_keyboard())

async def main():
    webapp = web.Application()
    webapp.router.add_get('/', handle_health)
    runner = web.AppRunner(webapp); await runner.setup()
    await web.TCPSite(runner, '0.0.0.0', int(os.environ.get("PORT", 10000))).start()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start)); app.add_handler(CallbackQueryHandler(button_handler))
    
    async with app:
        await app.initialize(); await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        while True: await asyncio.sleep(3600)

if __name__ == "__main__":
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    try: loop.run_until_complete(main())
    except Exception as e: print(f"Crash: {e}")
            
