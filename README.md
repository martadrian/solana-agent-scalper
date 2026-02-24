# Solana AI Agent Wallet
### Autonomous On-Chain Trading Agent

---

## 1. Overview
The Solana AI Agent Wallet is an autonomous software agent that combines a programmatic wallet with an AI decision engine to analyze markets and execute trades on-chain without manual intervention.

Unlike traditional wallets that act purely as signing tools, this system operates as an independent participant capable of:
* Generating and managing its own cryptographic identity.
* Interpreting live market data.
* Producing trading strategies via a Large Language Model (LLM).
* Executing token swaps through decentralized liquidity.
* Managing position lifecycle with automated exits.

The project serves as a prototype for agent-driven finance, where wallets evolve into active decision-making entities.

---

## 2. Core Capabilities

### 2.1 Autonomous Strategy Generation
The agent continuously evaluates market snapshots and requests a structured decision from an AI model. The response includes:
* **Action:** (BUY, SELL, WAIT)
* **Take-Profit:** Dynamic percentage based on volatility.
* **Stop-Loss:** Dynamic protection threshold.
* **Confidence Score:** 0-100% metric used to determine dynamic position sizing.

### 2.2 On-Chain Execution
All trades are executed through Jupiter liquidity on Solana Devnet. The agent:
1. Builds swap transactions via Versioned Transactions (V0).
2. Signs them locally using the resident private key.
3. Broadcasts via RPC to the Solana blockchain.
4. Verifies transaction signatures for total transparency.

### 2.3 Position Management
Active positions are tracked internally. When exit conditions (TP/SL) are met, the agent triggers the swap, records the trade in history, and clears the internal state.

### 2.4 Persistent Identity
Each user session maintains a deterministic wallet stored locally. On restart, the agent restores the same identity, public key, and balances.

### 2.5 Telegram Control Interface
| Command | Description |
| :--- | :--- |
| Start Agent | Initiates the autonomous trading loop. |
| Stop | Halts execution immediately. |
| Wallet | Displays the public key and current SOL balance. |
| History | Shows a log of the last 10 trades and signatures. |

---

## 3. Architectural Model

1. **Identity Layer:** Handles key generation, persistence, and local signing.
2. **Market Data Layer:** Fetches pricing and depth from Jupiter and Raydium APIs.
3. **Decision Layer:** Processes snapshots through an LLM to produce structured strategies.
4. **Execution Layer:** Constructs and submits real-time swaps.
5. **Interface Layer:** Provides real-time interaction via Telegram.

---

## 4. Trading Lifecycle
1. **Scan:** Market data is collected across configured pairs (SOL, RAY, USDC).
2. **Reason:** Snapshot is passed to the AI model for strategy generation.
3. **Plan:** Model returns a decision and specific risk parameters.
4. **Act:** If conditions are met, a real swap is executed on-chain.
5. **Monitor:** Continuous evaluation of TP and SL exit triggers.
6. **Close:** Position is closed automatically when thresholds are hit.

---

## 5. Security Considerations
* **Local Keys:** Private keys never leave the execution environment.
* **Non-Custodial:** No browser or external wallet integrations.
* **Stateless RPC:** Secure communication with blockchain nodes.
* **Headless:** Optimized for deployment on platforms like Render or AWS.

---

## 6. Environment Scope
The current implementation operates exclusively on:
* **Network:** Solana Devnet
* **Liquidity:** Jupiter Devnet Liquidity Pools
* **Risk:** Zero financial exposure using Devnet SOL.

---

## 7. Installation
```bash
git clone [https://github.com/martadrian/solana-agent-wallet.git](https://github.com/martadrian/solana-agent-wallet.git)
cd solana-agent-wallet
pip install -r requirements.txt
