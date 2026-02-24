---

name: solana-ai-agent-wallet
description: Autonomous AI trading wallet executing real Devnet swaps with LLM strategy generation, dynamic risk sizing, and on-chain execution logging.
version: 2.0.0

Solana AI Agent Wallet Skill

Overview

This skill defines an autonomous AI-powered participant within the Solana ecosystem.
The agent combines programmatic wallet infrastructure with real-time market intelligence and LLM-based reasoning to execute token swaps, manage risk, and maintain full on-chain transparency.

Operating entirely on Solana Devnet, the agent performs independent market evaluation, strategy synthesis, transaction signing, and lifecycle management without manual intervention.

---

Capabilities

- Persistent Programmatic Identity
  Automatically initializes or restores a deterministic Solana Keypair using local disk storage ("wallet_{chat_id}.json"), ensuring continuity across deployments.

- AI Strategy Engine
  Generates structured trading decisions ("BUY", "SELL", "WAIT") using live market snapshots processed through a Large Language Model.

- Confidence-Weighted Position Sizing
  Dynamically calculates trade allocation based on model confidence scores, enabling adaptive capital exposure.

- Real On-Chain Swap Execution
  Constructs and signs Versioned Transactions and executes swaps via Jupiter Devnet liquidity.

- Automated Risk Controls
  Implements dynamic Take-Profit and Stop-Loss thresholds per position with continuous monitoring and autonomous exit execution.

- Transparent Execution Logging
  Records transaction signatures, trade pairs, timestamps, and execution states with direct Devnet explorer verification links.

---

Interaction & Dashboard Triggers

The agent exposes the following operational controls via a persistent Telegram interface:

- "/start" — Initializes the agent environment and confirms wallet identity
- "run" — Activates the continuous market scanning and execution loop
- "stop" — Halts new trades while preserving position monitoring
- "wallet" — Displays programmatic address and real-time SOL balance
- "history" — Returns recent trade logs including signatures and pairs

---

Technical Specifications

- Network: Solana Devnet ("api.devnet.solana.com")
- Liquidity Source: Jupiter Aggregator (Devnet)
- Execution Model: Async event-driven Python loop
- Transaction Format: Versioned Transactions (V0)
- AI Layer: LLM-based structured decision engine
- Identity Security: Non-custodial local private key persistence
- Interface Layer: Telegram Bot API with inline control surface

---

Behavioral Model

The agent follows a continuous decision cycle:

1. Collect market snapshots across configured token pairs
2. Submit structured context to the AI decision engine
3. Receive strategy output with risk parameters
4. Execute swap if conditions and balance requirements are satisfied
5. Monitor active positions for TP/SL triggers
6. Log outcomes and update internal state

---

Security & Operational Design

- Local private keys never leave the execution environment
- Transactions are signed client-side prior to broadcast
- RPC communication remains stateless and read-verified
- Designed for headless cloud deployment (Render, VPS, containerized environments)

---

Scope

This skill operates exclusively within a simulated economic environment:

- Network: Solana Devnet
- Capital: Devnet SOL only
- Purpose: Autonomous agent research, testing, and demonstration

---

Summary

The Solana AI Agent Wallet skill represents a transition from passive wallet infrastructure to autonomous financial agents — combining AI reasoning, blockchain execution, and continuous risk management into a unified on-chain participant.
