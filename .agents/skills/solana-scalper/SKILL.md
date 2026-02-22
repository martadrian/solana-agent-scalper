---
name: solana-scalper
description: Autonomous Solana trading agent with persistent JSON wallet identity and 60-second time-decay exit logic.
version: 1.1.0
---

# Solana Scalper Skill

## Overview
This skill provides an autonomous decision engine for high-frequency scalping on Solana Devnet. It utilizes a persistent programmatic wallet identity and manages trade lifecycles through precise temporal windows.

## Capabilities
- **Persistent Wallet Management**: Automatically creates or recovers a unique Solana identity via `wallet_{chat_id}.json`.
- **Parallel Mesh Hunter**: Simultaneously monitors 10+ SPL token pairs (SOL, JUP, RAY, etc.) via Raydium V3 APIs.
- **4s-Shift Entry**: Triggers buy orders autonomously after detecting 4 consecutive 1-second price decreases.
- **Dual-Window Profit Guardian**:
    - **Window 1 (0-30s)**: Aggressive profit taking to cover 0.002 SOL network fees (Trigger: >2.1% gain).
    - **Window 2 (30-60s)**: Recovery mode to exit at the first sign of green (Break-even).
    - **Hard Exit (60s)**: Immediate liquidation to preserve capital.
- **Agentic Transaction Signing**: Native use of `solders` and `solana-py` for non-interactive transaction signing.

## Interaction & Commands
The agent is controlled via a Telegram interface using the following triggers:
- `/start`: Initializes the agentic environment and loads/creates the persistent wallet.
- `run` (Callback): Activates the autonomous parallel hunting and scalping loop.
- `stop` (Callback): Safely terminates the active hunting task.
- `wallet` (Callback): Provides live on-chain SOL balance and the public address.
- `history` (Callback): Displays the most recent trade executions and entry/exit prices.

## Technical Specifications
- **Network**: Solana Devnet (`api.devnet.solana.com`)
- **Protocol**: Raydium API + Solana System Program (Transfers)
- **Programming Language**: Python 3.10+
- **Security**: Environment-isolated credentials and local-only private key storage.
- 
