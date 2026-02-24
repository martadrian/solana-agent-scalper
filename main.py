import os, asyncio, nest_asyncio, logging, json, base64, httpx
from datetime import datetime
from aiohttp import web
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
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

# DevNet token mints
TOKENS = {
    "SOL": "So11111111111111111111111111111111111111112",
    "RAY": "RAYCPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C",
    "USDC": "USDCDevnetMint11111111111111111111111111111"
}

PAIRS = [("SOL", "RAY"), ("SOL", "USDC"), ("RAY", "USDC")]

JUPITER_QUOTE_API = "https://quote-api.jup.ag/v4/quote"
JUPITER_SWAP_API = "https://quote-api.jup.ag/v4/swap"
JUPITER_MARKET_API = "https://quote-api.jup.ag/v4/markets"

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
        self.active_positions = []  # track positions with entry, TP, SL

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
        except:
            return 0.0

    async def fetch_order_book(self, base, quote):
        """Fetch full depth from Jupiter DevNet"""
        async with httpx.AsyncClient() as client:
            params = {"inputMint": TOKENS[base], "outputMint": TOKENS[quote], "amount": str(int(0.1*1e9))}
            resp = await client.get(JUPITER_QUOTE_API, params=params)
            data = resp.json()
        price = float(data['data'][0]['outAmount'])/1e9 if data.get("data") else 0
        return {"pair": f"{base}/{quote}", "price": price}

    async def fetch_market_snapshots(self):
        """Scan multiple pairs and get depth/price info"""
        snapshots = []
        for base, quote in PAIRS:
            snapshot = await self.fetch_order_book(base, quote)
            snapshot["balance"] = await self.get_balance()
            snapshot["in_position"] = len([p for p in self.active_positions if p["pair"]==snapshot["pair"]])>0
            snapshots.append(snapshot)
        return snapshots

    async def generate_strategy(self, snapshot):
        """AI decides BUY/SELL/WAIT + TP/SL dynamically based on confidence"""
        prompt = (f"Market snapshot: {json.dumps(snapshot)}. "
                  f"Return ONLY JSON with keys: "
                  f"'strategy', 'action' ('BUY','SELL','WAIT'), "
                  f"'tp_pct', 'sl_pct', 'confidence' 0-100")
        for model in MODEL_LIST:
            try:
                res = or_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=15
                )
                return json.loads(res.choices[0].message.content)
            except:
                continue
        return None

    async def execute_actual_swap(self, action, base, quote, amount):
        """Swap tokens via Jupiter DevNet"""
        in_mint = TOKENS[base] if action=="BUY" else TOKENS[quote]
        out_mint = TOKENS[quote] if action=="BUY" else TOKENS[base]

        async with httpx.AsyncClient() as client:
            params = {
                "userPublicKey": str(self.keypair.pubkey()),
                "inputMint": in_mint,
                "outputMint": out_mint,
                "amount": str(int(amount*1e9)),
                "slippageBps": "100",
                "wrapUnwrapSOL": True
            }
            resp = await client.post(JUPITER_SWAP_API, json=params)
            data = resp.json()
            if not data.get("swapTransaction"):
                logging.error("Swap API error")
                return None
            raw_tx = base64.b64decode(data["swapTransaction"])
            tx = VersionedTransaction.from_bytes(raw_tx)
            recent_bh = await self.client.get_latest_blockhash()
            tx.message.recent_blockhash = recent_bh.value.blockhash
            tx.sign([self.keypair])
            sig = await self.client.send_raw_transaction(tx.serialize())
            return sig.value

    async def check_positions(self, snapshot, bot):
        """Check active positions for TP/SL for all pairs"""
        for pos in list(self.active_positions):
            current_price = snapshot["price"]
            if current_price >= pos["tp"]:
                sig = await self.execute_actual_swap("SELL", pos["pair"].split("/")[0], pos["pair"].split("/")[1], pos["amount"])
                if sig:
                    self.history.append(f"{datetime.now().strftime('%H:%M')} | TP SELL | {pos['pair']} | Tx: {sig[:8]}...")
                    await bot.send_message(self.chat_id, f"✅ **TP Hit SELL** {pos['pair']}\nTx: `https://solscan.io/tx/{sig}?cluster=devnet`", parse_mode="Markdown")
                    self.active_positions.remove(pos)
            elif current_price <= pos["sl"]:
                sig = await self.execute_actual_swap("SELL", pos["pair"].split("/")[0], pos["pair"].split("/")[1], pos["amount"])
                if sig:
                    self.history.append(f"{datetime.now().strftime('%H:%M')} | SL SELL | {pos['pair']} | Tx: {sig[:8]}...")
                    await bot.send_message(self.chat_id, f"❌ **SL Hit SELL** {pos['pair']}\nTx: `https://solscan.io/tx/{sig}?cluster=devnet`", parse_mode="Markdown")
                    self.active_positions.remove(pos)

# ================= LOOP =================
async def agent_loop(chat_id, bot):
    agent = manager.get_agent(chat_id)
    await bot.send_message(chat_id, "🚀 **Live Agent Started**. Trading on DevNet...", parse_mode="Markdown")

    while agent.is_running:
        snapshots = await agent.fetch_market_snapshots()

        for snapshot in snapshots:
            await agent.check_positions(snapshot, bot)
            strategy = await agent.generate_strategy(snapshot)

            if strategy:
                action = strategy.get("action")
                tp_pct = strategy.get("tp_pct", 2)
                sl_pct = strategy.get("sl_pct", 1)
                confidence = strategy.get("confidence", 50)
                trade_amount = round(0.1 * (confidence/100), 4)  # dynamic sizing

                if action=="BUY" and snapshot["balance"]>=trade_amount:
                    sig = await agent.execute_actual_swap("BUY", snapshot["pair"].split("/")[0], snapshot["pair"].split("/")[1], trade_amount)
                    if sig:
                        agent.active_positions.append({
                            "pair": snapshot["pair"],
                            "entry_price": snapshot["price"],
                            "amount": trade_amount,
                            "tp": snapshot["price"]*(1+tp_pct/100),
                            "sl": snapshot["price"]*(1-sl_pct/100)
                        })
                        agent.history.append(f"{datetime.now().strftime('%H:%M')} | BUY | {snapshot['pair']} | Tx: {sig[:8]}...")
                        await bot.send_message(chat_id, f"✅ **BUY executed** {snapshot['pair']}\nTx: `https://solscan.io/tx/{sig}?cluster=devnet`", parse_mode="Markdown")

        await asyncio.sleep(30)

# ================= MANAGER & TELEGRAM =================
class AgentManager:
    def __init__(self): self.users = {}
    def get_agent(self, chat_id):
        if chat_id not in self.users: self.users[chat_id] = SolanaAgent(chat_id)
        return self.users[chat_id]

manager = AgentManager()

async def button_handler(update, context):
    q = update.callback_query
    await q.answer()
    agent = manager.get_agent(q.message.chat_id)

    if q.data=="run" and not agent.is_running:
        agent.is_running = True
        asyncio.create_task(agent_loop(q.message.chat_id, context.bot))
    elif q.data=="stop":
        agent.is_running = False
        await q.message.reply_text("🛑 Agent Stopped", reply_markup=main_menu_keyboard())
    elif q.data=="wallet":
        bal = await agent.get_balance()
        await q.message.reply_text(f"💼 **Wallet**: `{agent.keypair.pubkey()}`\nBalance: {bal:.4f} SOL", parse_mode="Markdown")
    elif q.data=="history":
        h = "\n".join(agent.history[-10:]) if agent.history else "No history"
        await q.message.reply_text(f"📜 **History**:\n{h}", parse_mode="Markdown")

async def main():
    webapp = web.Application()
    webapp.router.add_get('/', lambda r: web.Response(text="Running"))
    runner = web.AppRunner(webapp)
    await runner.setup()
    await web.TCPSite(runner,'0.0.0.0',int(os.environ.get("PORT",10000))).start()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", lambda u,c: u.message.reply_text("🤖 Agent Ready", reply_markup=main_menu_keyboard())))
    app.add_handler(CallbackQueryHandler(button_handler))

    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        while True:
            await asyncio.sleep(3600)

if __name__=="__main__":
    asyncio.run(main())
