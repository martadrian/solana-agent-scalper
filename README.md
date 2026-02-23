# 🤖 Solana Agentic Wallet Prototype
**Autonomous Identity & High-Frequency Mesh Scalper**

This repository features a fully autonomous Agentic Wallet designed for the Solana ecosystem. Unlike traditional custodial or browser-based wallets, this system operates as an independent participant, capable of programmatic transaction signing, asset management, and complex protocol interaction without human intervention.

## 🚀 Core Capabilities
* **Programmatic Identity**: Automatically generates and recovers cryptographic Keypairs using `solders`, ensuring a unique and persistent on-chain identity.
* **Autonomous Signing Engine**: Signs and broadcasts transactions natively, enabling the agent to act on market opportunities in milliseconds.
* **Giga-Balance Optimization**: Specifically tuned for capital efficiency with **5 SOL+** balances, utilizing dynamic priority fees to ensure execution during network congestion.
* **Multi-Asset Management**: Native support for holding, tracking, and swapping between SOL and a broad mesh of SPL tokens (JUP, RAY, PYTH, etc.).

---

## 🧠 The "Profit Guardian" Strategy
The agent employs a sophisticated dual-window temporal exit strategy to solve the "bag-holding" problem in volatile markets:

| Phase | Window | Condition | Logic |
| :--- | :--- | :--- | :--- |
| **Hunting** | Continuous | 4-Drop Shift | Scans top volume pairs for 4 consecutive price drops. |
| **Window 1** | 0 - 180s | Target > 1.2% | Seeks a high-probability "Sniper" profit target. |
| **Window 2** | 180 - 240s | Price > Entry | **Recovery Mode**: Exits at the first sign of green to protect principal. |
| **Hard Exit** | 240s | Immediate | Full liquidation of position to reset the hunting cycle. |



---

## 🛠️ Architecture & Setup

### 1. Key Persistence & Disk Management
For the agent to maintain its identity across sessions, it utilizes a local filesystem-based storage system. 
* **Identity Restoration**: On boot, the agent scans for `wallet_{chat_id}.json`. If found, it restores the previous wallet; otherwise, it initializes a new one.
* **Deployment Note**: When deploying to cloud environments like **Render**, attach a **Persistent Disk** to the root directory. This ensures the agent’s funds and identity survive service restarts or redeployments.
* Telegram Interface
The agent is managed via an interactive dashboard:
🚀 Start Scalping: Engages the autonomous mesh hunter using 10% of current balance per trade.
🛑 Stop Agent: Gracefully terminates the loop after managing any active positions.
💼 Wallet: Real-time on-chain query of SOL balance and programmatic public key.
📜 Swap History: Detailed audit logs formatted as: Timestamp | Side | Pair | Amount SOL @ Price.
🛡️ Security & Design Considerations
Sandboxed Environment: Private keys are generated and stored locally within the execution environment and are never transmitted.
Non-Interactive Execution: Built for headless environments where "Connect Wallet" prompts are impossible.
Priority Fees: Implements set_compute_unit_price at 150,000 micro-lamports to ensure agent transactions are prioritized by the network.

### 2. Installation
```bash
# Clone the repository
git clone [https://github.com/martadrian/solana-agent-wallet.git](https://github.com/martadrian/solana-agent-wallet.git)
cd solana-agent-wallet

# Install dependencies
pip install -r requirements.txt

# Configure environment
echo "TELEGRAM_TOKEN=your_token_here" > .env
python bot.py
