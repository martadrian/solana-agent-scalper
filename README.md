# 🤖 Solana Agentic Mesh Scalper
**The Time-Windowed Autonomous Trading Agent**

This repository features a fully autonomous, agentic wallet designed for the Solana Devnet. It doesn't just execute trades; it "guards" them using a sophisticated dual-window time strategy to ensure maximum capital efficiency and profit protection.



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
* **Self-Aware Balance**: Monitors its own SOL and SPL token balances via the `AsyncClient` to determine trade viability before sending instructions.

### 2. Security Considerations
* **Non-Custodial Logic**: Private keys are generated and stored only within the execution environment and are never transmitted.
* **Environment Isolation**: Uses `os.getenv` to keep sensitive credentials (like your Telegram Token) out of the source code.
* **Risk Mitigation**: The 60-second hard exit prevents "bag holding" and protects the 0.1 SOL trade size from long-term downtrends.
* **Fee Awareness**: The agent is programmed with a "Profit Margin Floor" (~0.002 SOL) to ensure transaction fees do not erode the principal.

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

# Create a .env file for your credentials
echo "TELEGRAM_TOKEN=your_telegram_bot_token_here" > .env
echo "RPC_URL=[https://api.devnet.solana.com](https://api.devnet.solana.com)" >> .env
3. Requirements File (requirements.txt)
Ensure your requirements.txt contains the following:
python-telegram-bot
httpx
solana
solders
nest_asyncio
python-dotenv
4. Agent Skills Manifest (SKILL.md)
To comply with agentic standards, this repo includes a machine-readable skill manifest located at:
/.agents/skills/solana-scalper/SKILL.md
This file allows other AI agents to discover this wallet's capabilities, including parallel monitoring and automated trade execution.
🚀 Execution
Start the agent and message it /start on Telegram to initialize your programmatic wallet:
