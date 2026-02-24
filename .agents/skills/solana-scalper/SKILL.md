---

name: solana-autonomous-trading-wallet
description: Autonomous AI trading agent performing multi-pair analysis, Jupiter-based swap execution, and automated position lifecycle management on Solana Devnet.
version: 1.1.0
author: martadrian
license: MIT
compatibility: Solana Devnet, Python 3.10+, OpenAI/OpenRouter API

Solana Autonomous Trading Skill

This skill enables an AI agent to function as a self-custodial portfolio operator on the Solana blockchain.
It establishes a continuous execution loop combining live market observation, AI-driven strategy generation, and on-chain transaction execution.

---

Core Capabilities

1. Market Awareness

- Multi-Pair Monitoring
  Continuously scans liquidity across key trading pairs: "SOL/RAY", "SOL/USDC", and "RAY/USDC".

- Real-Time Pricing
  Retrieves live quotes and routing data through Jupiter Aggregator APIs to maintain accurate market context.

---

2. Autonomous Decision Making

- Confidence-Weighted Position Sizing
  Trade allocation is dynamically adjusted according to the AI model’s confidence score (0–100%).

- Structured Strategy Output
  Produces machine-readable trade decisions containing:
  
  - "action" → BUY / SELL / WAIT
  - "tp_pct" → Take-Profit threshold
  - "sl_pct" → Stop-Loss threshold

---

3. On-Chain Execution

- Jupiter Swap Integration
  Builds and signs Versioned Transactions for real Devnet swaps via Jupiter liquidity routes.

- Automated Position Lifecycle
  Continuously monitors open positions and autonomously executes exit trades when TP or SL conditions are met.

---

How to Use This Skill

Triggers

Invoke this skill when:

- An autonomous trading session needs to be initiated
- Continuous monitoring of specified token pairs is required
- Capital should be managed with automated risk controls

---

Operational Flow

1. Initialization
   Load or create the deterministic wallet from "wallet_{chat_id}.json".

2. Execution Loop
   
   - Call "fetch_market_snapshots()" to obtain current price context
   - Run "check_positions()" to evaluate exit conditions
   - Submit data to "generate_strategy()" for AI reasoning

3. Trade Execution
   If the AI response returns "BUY" or "SELL", invoke "execute_actual_swap()" to sign and broadcast the transaction.

---

Security Controls

- Private Key Isolation
  Keys remain stored locally and are never exposed to the AI model or external services.

- Slippage Protection
  Default slippage tolerance set to 100 Bps (1%) to mitigate adverse execution.

- Environment Restriction
  Hard-coded to Solana Devnet to eliminate risk of real-fund exposure.

---

Technical Specifications

- Blockchain Network: Solana Devnet
- Liquidity Source: Jupiter Aggregator (v4)
- Execution Runtime: Python async event loop
- Interface Layer: Telegram Bot API
- Transaction Format: Versioned Transactions (V0)

---

Summary

This skill transforms a standard programmatic wallet into an autonomous trading entity capable of observing markets, reasoning over data, and executing on-chain actions independently. It is designed as a research and demonstration framework for AI-driven financial agents operating in decentralized environments.
