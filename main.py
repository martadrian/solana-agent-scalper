import os, asyncio, nest_asyncio, logging, json, base64, httpx, math, sys
from datetime import datetime, timedelta
from aiohttp import web
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
import requests

nest_asyncio.apply()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RPC_URL = os.getenv("RPC_URL", "https://api.devnet.solana.com")
PORT = int(os.getenv("PORT", 10000))
WALLET_SECRET = os.getenv("WALLET_SECRET")

if not TELEGRAM_TOKEN or not OPENROUTER_API_KEY:
    logging.error("Missing tokens!")
    sys.exit(1)

TOKENS = {
    "SOL": "So11111111111111111111111111111111111111112",
    "USDC": "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU"
}

JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_API = "https://quote-api.jup.ag/v6/swap"

def main_menu_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Start Agent", callback_data="run"),
         InlineKeyboardButton("Stop", callback_data="stop")],
        [InlineKeyboardButton("Wallet", callback_data="wallet"),
         InlineKeyboardButton("History", callback_data="history")],
        [InlineKeyboardButton("Status", callback_data="status")]
    ])

class AutonomousAgent:
    def __init__(self, chat_id):
        self.chat_id = chat_id
        self.keypair = self._load_or_create_wallet()
        self.client = AsyncClient(RPC_URL, commitment=Confirmed)
        self.is_running = False
        self.history = []
        self.active_positions = []
        self.price_history = []
        self.trade_history = []
        self.loop_count = 0
        
    def _load_or_create_wallet(self):
        if WALLET_SECRET:
            try:
                secret_bytes = base64.b64decode(WALLET_SECRET)
                return Keypair.from_bytes(secret_bytes)
            except:
                pass
        kp = Keypair()
        secret = base64.b64encode(bytes(kp)).decode()
        logging.info("NEW WALLET: " + str(kp.pubkey()))
        logging.info("SAVE THIS: " + secret)
        return kp

    async def get_balance(self):
        try:
            res = await self.client.get_balance(self.keypair.pubkey())
            return res.value / 1e9
        except:
            return 0.0

    async def fetch_market_data(self):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                params = {
                    "inputMint": TOKENS["SOL"],
                    "outputMint": TOKENS["USDC"],
                    "amount": str(int(0.01 * 1e9)),
                    "slippageBps": "100"
                }
                resp = await client.get(JUPITER_QUOTE_API, params=params)
                data = resp.json()
                
                if not data.get("data"):
                    return None
                
                current_price = int(data["data"][0]["outAmount"]) / 1e6
                
                self.price_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "price": current_price
                })
                if len(self.price_history) > 100:
                    self.price_history.pop(0)
                
                prices = [p["price"] for p in self.price_history]
                
                return {
                    "current_price": current_price,
                    "price_history": prices[-20:],
                    "volatility": self._calc_volatility(prices),
                    "trend": self._detect_trend(prices),
                    "balance": await self.get_balance(),
                    "active_positions": len(self.active_positions),
                    "last_trade_result": self._last_trade_result()
                }
        except Exception as e:
            logging.error("Market data error: " + str(e))
            return None

    def _calc_volatility(self, prices):
        if len(prices) < 2:
            return 0
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        return sum(r**2 for r in returns) / len(returns) ** 0.5 if returns else 0

    def _detect_trend(self, prices):
        if len(prices) < 10:
            return "UNKNOWN"
        recent = prices[-5:]
        older = prices[-10:-5]
        if recent[-1] > older[-1] * 1.02:
            return "UPTREND"
        elif recent[-1] < older[-1] * 0.98:
            return "DOWNTREND"
        return "SIDEWAYS"

    def _last_trade_result(self):
        if not self.trade_history:
            return "No previous trades"
        last = self.trade_history[-1]
        if "profit_loss" in last:
            return "Last trade P&L: " + str(last['profit_loss'])
        return "Last trade pending"

    async def ai_decision(self, market_data):
        context = {
            "role": "autonomous_crypto_trader",
            "objective": "Maximize profit through active trading",
            "constraints": {
                "max_position_size": 0.5,
                "min_confidence": 20,
                "pair": "SOL/USDC",
                "exchange": "Jupiter DEX on Solana DevNet"
            },
            "market_data": market_data,
            "memory": {
                "recent_trades": self.trade_history[-5:],
                "performance": self._calculate_performance()
            }
        }
        
        prompt = "You are an AUTONOMOUS crypto trading AI. Analyze the market and decide.\n\nCONTEXT: " + json.dumps(context, indent=2) + "\n\nDECISION FRAMEWORK:\n- You have FULL autonomy to BUY, SELL, or WAIT\n- No human rules or validation - you decide based on market analysis\n- Consider: price action, trends, volatility, patterns, support/resistance\n- Learn from your past trades in the memory\n\nRESPONSE FORMAT (JSON only):\n{\n  \"action\": \"BUY\" or \"SELL\" or \"WAIT\",\n  \"confidence\": 0-100,\n  \"amount_percent\": 10-100,\n  \"tp_percent\": number,\n  \"sl_percent\": number,\n  \"reasoning\": \"Detailed technical analysis of why you made this decision\",\n  \"risk_assessment\": \"Your evaluation of trade risk\"\n}\n\nMake your decision now:"

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": "Bearer " + OPENROUTER_API_KEY,
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://render.com"
                },
                json={
                    "model": "meta-llama/llama-3.1-8b-instruct",
                    "messages": [
                        {"role": "system", "content": "You are an expert autonomous crypto trader. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.8,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.status_code != 200:
                logging.error("AI API error: " + str(response.status_code))
                return None
            
            content = response.json()["choices"][0]["message"]["content"]
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            decision = json.loads(content.strip())
            logging.info("AI DECISION: " + str(decision))
            return decision
            
        except Exception as e:
            logging.error("AI decision error: " + str(e))
            return None

    def _calculate_performance(self):
        if not self.trade_history:
            return {"total_trades": 0, "win_rate": 0, "total_pnl": 0}
        
        closed_trades = [t for t in self.trade_history if "profit_loss" in t]
        if not closed_trades:
            return {"total_trades": len(self.trade_history), "win_rate": 0, "total_pnl": 0}
        
        wins = sum(1 for t in closed_trades if t["profit_loss"] > 0)
        total_pnl = sum(t["profit_loss"] for t in closed_trades)
        
        return {
            "total_trades": len(closed_trades),
            "win_rate": wins / len(closed_trades),
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(closed_trades)
        }

    async def execute_trade(self, decision, market_data):
        action = decision.get("action", "WAIT")
        confidence = decision.get("confidence", 0)
        
        if action == "WAIT" or confidence < 20:
            logging.info("AI decided to WAIT (confidence: " + str(confidence) + ")")
            return None
        
        balance = market_data["balance"]
        amount_percent = decision.get("amount_percent", 50)
        amount = balance * (amount_percent / 100)
        
        amount = min(amount, balance * 0.5)
        amount = max(amount, 0.01)
        
        if amount > balance:
            logging.warning("Insufficient balance for AI's desired trade")
            return None
        
        if action == "BUY":
            return await self._execute_buy(amount, decision, market_data)
        elif action == "SELL":
            return await self._execute_sell(decision)
        
        return None

    async def _execute_buy(self, amount, decision, market_data):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                quote_params = {
                    "inputMint": TOKENS["SOL"],
                    "outputMint": TOKENS["USDC"],
                    "amount": str(int(amount * 1e9)),
                    "slippageBps": "100"
                }
                quote_resp = await client.get(JUPITER_QUOTE_API, params=quote_params)
                quote_data = quote_resp.json()
                
                if not quote_data.get("data"):
                    return None
                
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
                
                tp = market_data["current_price"] * (1 + decision.get("tp_percent", 2) / 100)
                sl = market_data["current_price"] * (1 - decision.get("sl_percent", 2) / 100)
                
                position = {
                    "entry_price": market_data["current_price"],
                    "amount": amount,
                    "tp": tp,
                    "sl": sl,
                    "tx": sig.value,
                    "time": datetime.now(),
                    "ai_decision": decision
                }
                self.active_positions.append(position)
                
                self.trade_history.append({
                    "action": "BUY",
                    "price": market_data["current_price"],
                    "amount": amount,
                    "tx": sig.value,
                    "ai_reasoning": decision.get("reasoning", ""),
                    "timestamp": datetime.now().isoformat()
                })
                
                logging.info("BUY executed: " + str(amount) + " SOL at " + str(market_data['current_price']))
                return sig.value, position
                
        except Exception as e:
            logging.error("Buy execution error: " + str(e))
            return None

    async def _execute_sell(self, decision):
        if not self.active_positions:
            return None
        
        position = self.active_positions[0]
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                quote_params = {
                    "inputMint": TOKENS["USDC"],
                    "outputMint": TOKENS["SOL"],
                    "amount": str(int(position["amount"] * 1e9)),
                    "slippageBps": "100"
                }
                quote_resp = await client.get(JUPITER_QUOTE_API, params=quote_params)
                quote_data = quote_resp.json()
                
                if not quote_data.get("data"):
                    return None
                
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
                
                exit_price = position["amount"]
                pnl = (exit_price - position["entry_price"]) * position["amount"]
                
                self.trade_history.append({
                    "action": "SELL",
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "amount": position["amount"],
                    "profit_loss": pnl,
                    "tx": sig.value,
                    "ai_reasoning": decision.get("reasoning", ""),
                    "timestamp": datetime.now().isoformat()
                })
                
                self.active_positions.remove(position)
                
                logging.info("SELL executed: P&L " + str(pnl))
                return sig.value, pnl
                
        except Exception as e:
            logging.error("Sell execution error: " + str(e))
            return None

    async def check_positions(self, current_price, bot):
        for pos in list(self.active_positions):
            if current_price >= pos["tp"]:
                logging.info("Take profit hit: " + str(current_price) + " >= " + str(pos['tp']))
                await self._close_position(pos, "TP", current_price, bot)
            elif current_price <= pos["sl"]:
                logging.info("Stop loss hit: " + str(current_price) + " <= " + str(pos['sl']))
                await self._close_position(pos, "SL", current_price, bot)

    async def _close_position(self, position, reason, current_price, bot):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                quote_params = {
                    "inputMint": TOKENS["USDC"],
                    "outputMint": TOKENS["SOL"],
                    "amount": str(int(position["amount"] * 1e9)),
                    "slippageBps": "100"
                }
                quote_resp = await client.get(JUPITER_QUOTE_API, params=quote_params)
                quote_data = quote_resp.json()
                
                if not quote_data.get("data"):
                    return
                
                swap_body = {
                    "quoteResponse": quote_data,
                    "userPublicKey": str(self.keypair.pubkey()),
                    "wrapAndUnwrapSOL": True
                }
                
                swap_resp = await client.post(JUPITER_SWAP_API, json=swap_body)
                swap_data = swap_resp.json()
                
                if not swap_data.get("swapTransaction"):
                    return
                
                raw_tx = base64.b64decode(swap_data["swapTransaction"])
                tx = VersionedTransaction.from_bytes(raw_tx)
                
                recent_bh = await self.client.get_latest_blockhash()
                tx.message.recent_blockhash = recent_bh.value.blockhash
                tx.sign([self.keypair])
                
                sig = await self.client.send_raw_transaction(tx.serialize())
                
                pnl = (current_price - position["entry_price"]) * position["amount"]
                
                self.trade_history.append({
                    "action": "SELL",
                    "entry_price": position["entry_price"],
                    "exit_price": current_price,
                    "amount": position["amount"],
                    "profit_loss": pnl,
                    "tx": sig.value,
                    "close_reason": reason,
                    "timestamp": datetime.now().isoformat()
                })
                
                self.active_positions.remove(position)
                
                solscan = "https://solscan.io/tx/" + sig.value + "?cluster=devnet"
                
                message = "Position Closed (" + reason + ")\nP&L: " + str(pnl) + "\n[View on Solscan](" + solscan + ")"
                
                await bot.send_message(
                    self.chat_id,
                    message,
                    parse_mode="Markdown",
                    reply_markup=main_menu_keyboard()
                )
                
        except Exception as e:
            logging.error("Close position error: " + str(e))

async def autonomous_loop(chat_id, bot):
    agent = manager.get_agent(chat_id)
    
    await bot.send_message(
        chat_id,
        "AUTONOMOUS AGENT STARTED\n\nThe AI is now in control.\nIt will analyze the market and make its own decisions.\nNo human rules applied.",
        reply_markup=main_menu_keyboard()
    )
    
    while agent.is_running:
        try:
            agent.loop_count += 1
            logging.info("AUTONOMOUS LOOP #" + str(agent.loop_count))
            
            market_data = await agent.fetch_market_data()
            if not market_data:
                logging.error("Failed to fetch market data")
                await asyncio.sleep(30)
                continue
            
            logging.info("Price: " + str(market_data['current_price']) + ", Balance: " + str(market_data['balance']))
            
            decision = await agent.ai_decision(market_data)
            if not decision:
                logging.error("AI failed to decide")
                await asyncio.sleep(30)
                continue
            
            if decision["action"] in ["BUY", "SELL"]:
                result = await agent.execute_trade(decision, market_data)
                
                if result:
                    sig, details = result
                    solscan = "https://solscan.io/tx/" + sig + "?cluster=devnet"
                    
                    # Simple string concatenation instead of f-strings
                    action_str = decision['action']
                    conf_str = str(decision['confidence'])
                    reason_str = decision.get(
