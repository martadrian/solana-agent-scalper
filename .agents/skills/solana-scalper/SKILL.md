---
name: solana-agent-wallet
description: Autonomous Agentic Wallet with 4-minute temporal exit logic, dynamic 10% trade sizing, and priority fee execution.
version: 1.2.0
---

# Solana Agentic Wallet Skill

## Overview
This skill defines an autonomous participant for the Solana ecosystem. It is capable of independent market analysis, programmatic transaction signing, and sophisticated liquidity management using a time-decay exit strategy optimized for Devnet assets.

## Capabilities
- **Persistent Programmatic Identity**: Automatically initializes or restores a unique Solana Keypair via local disk storage (`wallet_{chat_id}.json`).
- **Autonomous Mesh Hunter**: Monitors high-volume token pairs (SOL, JUP, RAY, PYTH, etc.) using high-frequency price polling.
- **Dynamic Unit Economics**: Automatically calculates trade size as 10% of total wallet balance to enable compound growth.
- **Priority Execution Engine**: Implements `set_compute_unit_price` at 150,000 micro-lamports to ensure agent transactions bypass network congestion.
- **4-Minute Profit Guardian**:
    - **Sniper Window (0-180s)**: Targets a 1.2% net profit to capitalize on momentum.
    - **Recovery Window (180-240s)**: Exits immediately upon any price movement above the entry point.
    - **Liquidate Window (240s)**: Force-closes positions to free up capital for new opportunities.

## Interaction & Dashboard Triggers
The agent exposes the following autonomous functions via a persistent Telegram interface:
- `/start`: Provisions the agentic environment and verifies identity persistence.
- `run`: Activates the high-frequency scanning and auto-signing loop.
- `stop`: Deactivates hunting while maintaining management of open positions.
- `wallet`: Real-time query of programmatic address and SOL holdings.
- `history`: Provides detailed trade logs: `Time | Side | Pair | Amount @ Price`.

## Technical Specifications
- **Network**: Solana Devnet (`api.devnet.solana.com`)
- **Execution Environment**: Python 3.10+ / `solders` / `solana-py`
- **Protocol Integration**: Raydium V3 Price APIs & Solana System Program
- **Identity Security**: Non-custodial, local-only secret key storage with disk persistence support.
- 
