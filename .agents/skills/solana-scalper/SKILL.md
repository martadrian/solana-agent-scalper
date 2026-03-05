# Agentic Wallet - AI Agent Skills Reference

## System Overview
Autonomous trading agent for Solana with encrypted wallet persistence, AI-driven decisions, and real DEX integration. Built for devnet with production-grade security.

## Core Capabilities

Wallet Management:
- Create Solana wallets programmatically
- Encrypt and persist keys in Supabase
- Automatic key loading or generation
- Never log private keys to console
- Sign transactions autonomously

Market Analysis:
- Real-time price fetching from Orca Whirlpools
- Volatility calculation from price history
- Trend detection (UPTREND/DOWNTREND/SIDEWAYS)
- AI-powered decision making via Groq LLM
- Confidence scoring for all decisions

Trading Execution:
- Execute real swaps on Orca DEX
- Automated position sizing (max 50% of balance)
- Take-profit and stop-loss management
- Transaction simulation before execution
- Slippage protection (default 1%)

Risk Management:
- Maximum position limits
- Emergency exit capability
- Balance validation before trades
- Price anomaly detection
- Automatic retry with backoff

Monitoring and Control:
- Telegram bot interface for all operations
- Real-time trade notifications
- Performance metrics (PnL, win rate)
- Health check web API
- Multi-agent swarm support

## Architecture Patterns

Async/Await:
All I/O operations are async including RPC calls, API requests, and Telegram updates.

Encrypted Persistence:
Keys stored encrypted in Supabase. Decrypted on load, held in memory only.

Modular AI:
AIOracle class can swap between providers without changing business logic.

Retry Logic:
All external calls have exponential backoff retry.

## Key Classes

SupabaseKeyManager: Encrypted key storage
AIOracle: AI decision engine
OrcaWhirlpoolClient: DEX integration
SolanaClient: RPC wrapper
AgenticWallet: Core agent logic
MultiAgentSwarm: Multi-agent coordinator

## Environment Setup

Required: TELEGRAM_TOKEN, GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY
Optional: RPC_URL (default: https://api.devnet.solana.com), PORT

## Safety Rules

Never use mainnet - devnet only
Never log private keys
Always simulate before executing
Validate all prices before trading
Check balances before any transaction

## Common Operations

Start agent: Click Start Agent button in Telegram
Stop agent: Click Stop Agent button in Telegram
Check wallet: Click Wallet button in Telegram
View trades: Click History button in Telegram
Request funds: Click Airdrop button in Telegram

## Troubleshooting

No valid pools: Check RPC_URL and network status
AI decision error: Check GROQ_API_KEY and rate limits
Key integrity failed: Key corruption detected, regenerate
Insufficient balance: Request airdrop or fund wallet
