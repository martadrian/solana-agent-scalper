import os, asyncio, random, nest_asyncio, logging, json
from datetime import datetime
from aiohttp import web
from solders.keypair import Keypair
from solders.transaction import Transaction
from solders.system_program import TransferParams, transfer
from solders.compute_budget import set_compute_unit_price 
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler
from openai import OpenAI

# ================= CONFIG =================
nest_asyncio.apply()
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
RPC_URL = os.getenv("RPC_URL", "https://api.devnet.solana.com")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODEL_LIST = ["meta-llama/llama-3.3-70b-instruct:free", "openrouter/auto"]
or_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# ================= UI =================
def main_menu_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🚀 Start Agent", callback_data="run"),
         InlineKeyboardButton("🛑 Stop", callback_data="stop")],
        [InlineKeyboardButton("💼 Wallet", callback_data="wallet"),
         InlineKeyboardButton("📜 History", callback_data="history")]
    ])

# ================= AGENT =================
class SolanaAgent:
    def __init__(self, chat_id):
        self.chat_id = chat_id
        self.wallet_path = f"wallet_{chat_id}.json"
        self.keypair = self.load_or_create_keypair()
        self.client = AsyncClient(RPC_URL, commitment=Confirmed)
        self.is_running = False
        self.history = []
        self.active_trade = None
        self.priority_fee = 150000 

    def load_or_create_keypair(self):
        if os.path.exists(self.wallet_path):
            with open(self.wallet_path, "r") as f:
                return Keypair.from_bytes(bytes(json.load(f)))
        kp = Keypair()
        with open(self.wallet_path, "w") as f:
            json.dump(list(bytes(kp)), f)
        return kp

    async def get_balance(self):
        try:
            res = await self.client.get_balance(self.keypair.pubkey())
            return res.value / 1e9
        except: return 0.0

    async def fetch_market_snapshot(self):
        # Mock snapshot for DevNet
        return {
            "pair": "SOL/USDC",
            "price": 105.42 + random.uniform(-0.5, 0.5),
            "volatility": "High",
            "trend": "Bullish",
            "balance": await self.get_balance()
        }

    async def generate_strategy(self, snapshot):
        prompt = f"Market: {json.dumps(snapshot)}. Return ONLY JSON with keys: strategy, action (BUY/WAIT), tp_pct, sl_pct, confidence."
        for model in MODEL_LIST:
            try:
                res = or_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=15
                )
                return json.loads(res.choices[0].message.content)
            except: continue
        return None

    async def execute_on_chain_trade(self, strategy, snapshot):
        try:
            trade_size = snapshot["balance"] * 0.10
            if trade_size < 0.001: return False

            # On-chain mock transfer (self-transfer for prototype)
            recent_blockhash = await self.client.get_latest_blockhash()
            ix_priority = set_compute_unit_price(self.priority_fee)
            ix_trade = transfer(TransferParams(
                from_pubkey=self.keypair.pubkey(),
                to_pubkey=self.keypair.pubkey(),
                lamports=int(0.000005 * 1e9)
            ))

            tx = Transaction.new_signed_with_payer(
                [ix_priority, ix_trade], 
                self.keypair.pubkey(), [self.keypair], 
                recent_blockhash.value.blockhash
            )
            await self.client.send_transaction(tx)

            self.history.append(
                f"{datetime.now().strftime('%H:%M')} | {strategy['action']} {snapshot['pair']} | Size: {trade_size:.3f} SOL | Entry: ${snapshot['price']:.2f} | Strategy: {strategy['strategy']} | Confidence: {strategy['confidence']}%"
            )
            return trade_size
        except Exception as e:
            logging.error(f"Trade Error: {e}")
            return False

# ================= LOOP =================
async def agent_loop(chat_id, bot):
    agent = manager.get_agent(chat_id)
    await bot.send_message(chat_id, "🧠 AI Brain Online — analyzing market mesh...", reply_markup=main_menu_keyboard())

    while agent.is_running:
        snapshot = await agent.fetch_market_snapshot()
        strategy = await agent.generate_strategy(snapshot)

        if strategy and strategy.get("action") == "BUY":
            size = await agent.execute_on_chain_trade(strategy, snapshot)
            if size:
                await bot.send_message(
                    chat_id,
                    f"🎯 Trade Executed\nStrategy: {strategy['strategy']}\nPair: {snapshot['pair']}\nSize: {size:.3f} SOL\nEntry: ${snapshot['price']:.2f}\nConfidence: {strategy['confidence']}%",
                    reply_markup=main_menu_keyboard()
                )
        await asyncio.sleep(30)

# ================= MANAGER =================
class AgentManager:
    def __init__(self): self.users = {}
    def get_agent(self, chat_id):
        if chat_id not in self.users: self.users[chat_id] = SolanaAgent(chat_id)
        return self.users[chat_id]

manager = AgentManager()

# ================= TELEGRAM =================
async def button_handler(update, context):
    q = update.callback_query
    await q.answer()
    agent = manager.get_agent(q.message.chat_id)

    if q.data == "run" and not agent.is_running:
        agent.is_running = True
        asyncio.create_task(agent_loop(q.message.chat_id, context.bot))
    elif q.data == "stop":
        agent.is_running = False
        await q.message.reply_text("🛑 Agent Stopped", reply_markup=main_menu_keyboard())
    elif q.data == "wallet":
        bal = await agent.get_balance()
        await q.message.reply_text(f"💼 Balance: {bal:.4f} SOL\nAddress: `{agent.keypair.pubkey()}`", reply_markup=main_menu_keyboard(), parse_mode="Markdown")
    elif q.data == "history":
        h = "\n".join(agent.history[-5:]) if agent.history else "No trades yet"
        await q.message.reply_text(f"📜 History:\n{h}", reply_markup=main_menu_keyboard())

async def main():
    webapp = web.Application()
    webapp.router.add_get('/', lambda r: web.Response(text="Agent Live"))

    runner = web.AppRunner(webapp)
    await runner.setup()
    await web.TCPSite(runner, '0.0.0.0', int(os.environ.get("PORT", 10000))).start()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("🤖 AI Agent Ready", reply_markup=main_menu_keyboard())))
    app.add_handler(CallbackQueryHandler(button_handler))

    async with app:
        await app.initialize(); await app.start()
        await app.updater.start_polling()
        while True: await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
