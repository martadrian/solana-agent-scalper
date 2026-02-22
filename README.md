# 🤖 Solana Agentic Mesh Scalper
**The Time-Windowed Autonomous Trading Agent**

This repository features a fully autonomous, agentic wallet designed for the Solana Devnet. It doesn't just execute trades; it "guards" them using a sophisticated dual-window time strategy to ensure maximum capital efficiency and profit protection.

## 🌟 Live Demo
Experience the autonomous agent in action on Telegram:
👉 **[@Aiagentscalperbot](https://t.me/Aiagentscalperbot)**

---

## 🎮 Telegram Interface & Functionality
The agent is managed entirely through an interactive Telegram menu. Once you send `/start`, the agent initializes your unique persistent wallet and presents the following dashboard:

### 🛠️ Menu Buttons & Logic
- **🚀 Start Scalping**: 
  - *Internal Trigger:* `run`
  - *Action:* Activates the autonomous "Mesh Hunter." The agent begins scanning the top 10 token pairs every second. It looks for 4 consecutive 1-second price drops to execute a high-probability entry.
- **🛑 Stop Agent**: 
  - *Action:* Gracefully terminates the hunting loop. If a trade is currently active, the agent will continue to manage the exit strategy before stopping to protect your capital.
- **💼 Wallet**: 
  - *Action:* On-chain query to display your live **Devnet SOL Balance** and your unique **Programmatic Public Key**.
- **📜 Swap History**: 
  - *Action:* Pulls the last 5 trades from the local history, showing the token pair, buy price, sell price, and the exact timestamp of execution.

---

## 🌟 Key Features
- **Parallel Mesh Hunter**: Simultaneously monitors the top 10 SPL token pairs (SOL, JUP, RAY, etc.) for rapid momentum shifts.
- **4s-Shift Detection**: Uses a high-frequency tracking registry to identify 4 consecutive 1-second price drops—a high-probability scalping entry.
- **Agentic Decision Engine**: Independently manages the trade lifecycle without human oversight, adhering to strict time-based exit rules.
- **Devnet Ready**: Pre-configured for seamless testing on Solana Devnet with standard RPC integrations.

---

## 🧠 Deep Dive: Wallet Design & Strategy

### 1. Programmatic Wallet Design
The agent utilizes the `solders` library to create and manage an on-chain identity. Unlike a standard browser wallet, this **Agentic Wallet**:
* **Autonomous Generation**: Generates its own cryptographic Keypair programmatically upon initialization (`Keypair()`).
* **Non-Interactive Signing**: Signs transactions automatically within the execution loop using the `solders.transaction.Transaction` class.
* **Identity Persistence**: Saves the secret key locally in a `wallet_{chat_id}.json` file. This ensures your wallet address and funds remain consistent even if the server restarts.

### 2. Security Considerations
* **Non-Custodial Logic**: Private keys are generated and stored only within the execution environment and are never transmitted.
* **Environment Isolation**: Uses `os.getenv` to keep sensitive credentials (like your Telegram Token) out of the source code.
* **Risk Mitigation**: The 60-second hard exit prevents "bag holding" and protects the 0.1 SOL trade size from long-term downtrends.
* **Fee Awareness**: The agent is programmed with a "Profit Margin Floor" (~0.002 SOL / 2.1%) to ensure transaction fees do not erode the principal.

### 3. The "Profit Guardian" Strategy
The agent follows a strict temporal logic to solve the "bag-holding" problem in volatile markets:

| Phase | Duration | Exit Condition | Goal |
| :--- | :--- | :--- | :--- |
| **Hunting Phase** | Continuous | 4s Consecutive Price Drop | Find high-momentum entries. |
| **Window 1** | 0 - 30s | Profit > 2.1% OR Price > Buy @ 30s | Secure quick gains & cover fees. |
| **Window 2** | 30 - 60s | Price > Buy Price | Exit at the first second of profit. |
| **Hard Exit** | 60s | Immediate Sell | Liquidate position to restart hunting. |



---

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.10+
- A Telegram Bot Token from [@BotFather](https://t.me/botfather)
- A Solana Devnet RPC URL (Default: `https://api.devnet.solana.com`)

### 2. Installation & Setup
To get the agent running locally or on a server:

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/solana-agent-scalper.git](https://github.com/YOUR_USERNAME/solana-agent-scalper.git)
cd solana-agent-scalper

# Install required Python dependencies
pip install -r requirements.txt

# Create a .env file for your credentials (FOR LOCAL TESTING ONLY)
echo "TELEGRAM_TOKEN=your_telegram_bot_token_here" > .env
echo "RPC_URL=[https://api.devnet.solana.com](https://api.devnet.solana.com)" >> .env
