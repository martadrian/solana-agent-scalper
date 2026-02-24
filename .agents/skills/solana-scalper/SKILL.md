---
name: solana-autonomous-trading-wallet
description: "Autonomous AI trading agent performing multi-pair analysis, Jupiter-based swap execution, and automated position lifecycle management on Solana Devnet."
version: 1.1.0
author: martadrian
license: MIT
compatibility: "Solana Devnet, Python 3.10+, OpenAI/OpenRouter API"
---
Solana Autonomous Trading Skill
This skill allows an AI agent to operate as a self-custodial fund manager on the Solana blockchain. It provides a complete loop for market monitoring, AI-based decision making, and automated trade execution.
Core Capabilities
1. Market Awareness
Multi-Pair Scanning: Monitors token pairs SOL/RAY, SOL/USDC, and RAY/USDC.
Order Book Fetching: Retrieves real-time pricing and depth via Jupiter Quote API.
2. Autonomous Decision Making
Confidence-Weighted Sizing: Dynamically sizes trades based on AI confidence (0–100%).
Structured Strategy Output: Generates JSON with keys:
action (BUY, SELL, WAIT)
tp_pct (Take-Profit %)
sl_pct (Stop-Loss %)
confidence (0–100)
3. On-Chain Execution
Jupiter Swap Integration: Executes swaps via Versioned Transactions (V0) on Devnet.
Automated Lifecycle Management: Monitors active positions to trigger Take-Profit or Stop-Loss without manual input.
How to Use This Skill
Triggers
Use this skill when:
Starting an autonomous trading cycle on Solana Devnet.
Continuous monitoring of specific token pairs is required.
Automated risk management (TP/SL) is desired.
Operational Instructions
Initialization: Load or create the deterministic wallet from wallet_{chat_id}.json.
Market Loop:
fetch_market_snapshots() → retrieves current market prices.
check_positions() → evaluates active positions for exit triggers.
generate_strategy() → AI produces the next trade signal.
Execution: If action is BUY or SELL, call execute_actual_swap() to sign and broadcast the transaction.
Security Controls
Private Key Isolation: Keys remain local, never transmitted externally.
Slippage Protection: Locked at 100 Bps (1%) by default to prevent front-running.
Devnet Locked: Operates exclusively on Devnet to avoid real-fund risk.
Technical Specifications
Blockchain: Solana (Devnet)
DEX Aggregator: Jupiter v4
Interface: Telegram Bot API
Language: Python 3.10+
AI Integration: OpenAI / OpenRouter API
Key Management: Local deterministic wallet persistence (wallet_{chat_id}.json)
