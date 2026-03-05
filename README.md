# Autonomous Agentic Wallet System for Solana

A production-ready autonomous trading agent with full wallet control, encrypted key persistence, AI-driven market analysis, and real-time trade execution. Built for Solana devnet with Telegram integration for monitoring and control.

## Overview

This system creates autonomous economic agents that operate 24/7 without human intervention. Each agent has its own encrypted wallet, makes independent trading decisions using AI, and executes real transactions on Solana devnet.

Key capabilities:
- Create and manage Solana wallets programmatically with encrypted key storage in Supabase
- Analyze markets in real-time using Groqq LLM (Llama 3.3 70B) for decision making
- Execute actual trades automatically on Orca Whirlpools DEX with slippage protection
- Manage risk with configurable take-profit and stop-loss mechanisms
- Persist wallet keys securely across server restarts using Supabase encryption
- Monitor and control via Telegram bot interface with real-time notifications

## System Architecture

The system uses a modular async architecture with 7 core components:

1. SupabaseKeyManager: Encrypts private keys with AES-256, stores in Supabase database, never logs to console, survives server restarts, uses SHA-256 integrity verification
2. AIOracle: Groq LLM integration for market analysis, outputs BUY/SELL/WAIT/EMERGENCY_EXIT decisions with 0-100% confidence scores, JSON-structured responses, automatic fallback to rule-based logic if AI unavailable
3. OrcaWhirlpoolClient: Real DEX integration with Orca Whirlpools on devnet, live price fetching from multiple pools, automated swap execution with 1% default slippage, tick array calculation for concentrated liquidity, transaction simulation before execution
4. SolanaClient: Async RPC client with exponential backoff retry logic, SOL and SPL token balance monitoring, Associated Token Account (ATA) creation, devnet airdrop requests, transaction confirmation handling
5. AgenticWallet: Core autonomous trading engine with 60-second loop, position management with TP/SL triggers, portfolio tracking with realized and unrealized PnL, risk management with max 50% position sizing, win rate calculation
6. MultiAgentSwarm: Spawns multiple agents with isolated wallets, independent operation per agent, centralized Telegram monitoring, health check API for all agents
7. Telegram Bot Interface: Button-based control panel, real-time trade notifications, wallet inspection, performance statistics, airdrop requests

## Telegram Bot Workflow and Operation

The Telegram bot provides complete control over the autonomous agent. Users interact through buttons rather than typing commands.

Starting the bot:
User sends /start or clicks Start button. Bot displays main menu with 6 buttons arranged in 3 rows of 2 buttons each.

Main menu buttons:
Row 1: Start Agent (rocket emoji) and Stop Agent (stop sign emoji)
Row 2: Wallet (money bag emoji) and Status (chart emoji)
Row 3: Positions (chart increasing emoji) and History (scroll emoji)
Row 4: Airdrop (droplet emoji)

Button functions explained:

Start Agent button:
When clicked, the bot initializes the agent by creating or loading an encrypted wallet from Supabase, checking the SOL balance, loading Orca Whirlpool pools and selecting the best one by liquidity, starting the Groq AI client, and beginning the autonomous trading loop. The bot replies with the agent ID, wallet address, key storage type (Supabase persistent or memory-only warning), reminder that this executes real transactions on devnet, link to fund the wallet via faucet.solana.com, and a Solscan link to view the address.

Stop Agent button:
Immediately halts the autonomous trading loop. The bot confirms the agent has stopped. All positions remain open but no new trades will be executed.

Wallet button:
Displays current wallet information including the full Solana address (clickable), current SOL balance with 4 decimal precision, key storage type (Supabase persistent or memory volatile), and a Solscan link to view account details on devnet.

Status button:
Shows comprehensive performance metrics including agent ID and running status (running or stopped), key storage type, number of completed trading loops, count of active open positions, total profit and loss across all trades, win rate as percentage, and total number of trades executed.

Positions button:
Lists all currently open positions with entry price, position size in tokens, take profit price level, stop loss price level, and transaction signature link to Solscan. If no positions open, displays "No active positions."

History button:
Shows last 5 trades with action type (BUY/SELL), execution price, profit or loss amount, AI confidence percentage at time of trade, and transaction link. Uses green circle emoji for profits, red circle for losses, white circle for neutral.

Airdrop button:
Requests 2 SOL from the devnet faucet. Returns transaction signature if successful, or error message with link to manual faucet if failed.

Trading Loop Execution:
Once started, the agent runs continuously with this cycle every 60 seconds:
Fetches current price from best Orca Whirlpool pool, calculates 3-period and 10-period moving averages to determine trend (UPTREND if short MA exceeds long MA by 1%, DOWNTREND if below by 1%, otherwise SIDEWAYS), calculates volatility from standard deviation of returns, builds context object with current balances, open positions, total PnL, and win rate, sends market state and context to Groq AI for analysis, receives JSON decision with action type, confidence percentage, position size percentage, reasoning text, risk assessment, take profit percentage, and stop loss percentage, validates confidence meets minimum threshold of 20%, executes decision by opening position for BUY, closing position for SELL, doing nothing for WAIT, or closing all positions for EMERGENCY_EXIT, checks existing positions against take profit and stop loss levels and closes if triggered, logs all activity, sends Telegram notification if trade executed, waits 60 seconds before next iteration.

## Security Architecture

Private key protection:
Keys generated using cryptographically secure random number generator via Solders library. Immediately encrypted and stored in Supabase database using base64 encoding with row-level security. Never printed to console logs or error messages. Loaded into memory only when needed for transaction signing. SHA-256 hash verified on each use to detect corruption. Memory cleared on process exit.

Transaction safety:
All swap transactions simulated on RPC before execution to catch errors. Slippage tolerance set to 100 basis points (1%) to prevent failed trades. Compute budget instructions included to ensure transaction success. Price validation rejects extreme values outside reasonable bounds. Balance checks prevent transactions with insufficient funds.

Operational safety:
Devnet-only operation prevents accidental mainnet transactions. Emergency exit button allows instant liquidation of all positions. Automatic retry with exponential backoff for failed RPC calls. Position size limits prevent overexposure (max 50% of balance).

## Configuration Guide

Step 1: Create Telegram Bot
Open Telegram and search for @BotFather. Send /newbot command. Choose name for your bot. Choose username ending in bot. Copy the HTTP API token provided (starts with numbers followed by colon and random string). This is your TELEGRAM_TOKEN.

Step 2: Get Groq API Key
Visit console.groq.com and create free account. Navigate to API Keys section. Click Create API Key. Copy the key starting with gsk_. This is your GROQ_API_KEY. Free tier includes 20 requests per minute and 500,000 tokens per day.

Step 3: Setup Supabase Project
Go to supabase.com and create new project. Choose organization and project name. Set secure database password. Select region closest to your deployment. Wait for database to provision. From project dashboard, click Settings then API. Copy Project URL (format: https://xxxxxxxxxxxx.supabase.co). This is your SUPABASE_URL. Copy service_role secret (starts with eyJ). This is your SUPABASE_KEY. Go to Table Editor and create new table named wallet_keys with columns: agent_id (int8, primary key), encrypted_key (text), pubkey (text), updated_at (timestamptz).

Step 4: Environment Variables
Set these environment variables in your deployment platform (Render, Railway, or local machine):
TELEGRAM_TOKEN=your_telegram_bot_token_here
GROQ_API_KEY=your_groq_api_key_here
SUPABASE_URL=your_supabase_project_url_here
SUPABASE_KEY=your_supabase_service_role_key_here
RPC_URL=https://api.devnet.solana.com (optional, defaults to devnet)
PORT=10000 (optional, defaults to 10000)

Step 5: Installation Commands
Run these commands in terminal:
git clone https://github.com/yourusername/agentic-wallet.git
cd agentic-wallet
python -m venv venv
source venv/bin/activate (on Linux/Mac) or venv\Scripts\activate (on Windows)
pip install -r requirements.txt

Step 6: Running the System
With environment variables set, run:
python main.py

You should see log output showing web server starting on port 10000, Telegram bot starting if token valid, and ready message with key storage status.

Step 7: First Use
Open Telegram and find your bot by username. Click Start or send /start. Click Start Agent button. Bot will display your wallet address. Fund this address with devnet SOL from faucet.solana.com. Trading will begin automatically once balance detected.

## Technical Specifications

Trading Parameters:
MIN_CONFIDENCE = 20 (minimum AI confidence percentage required to execute trade, below this converts to WAIT)
MAX_POSITION_SIZE_PCT = 50 (maximum percentage of wallet balance to use in single trade)
DEFAULT_SLIPPAGE_BPS = 100 (slippage tolerance in basis points, 100 = 1%)
TICK_ARRAY_SIZE = 88 (Orca Whirlpools constant for tick array calculation)

AI Configuration:
Model: llama-3.3-70b-versatile (Groq hosted)
Temperature: 0.3 (low randomness for consistent decisions)
Max tokens: 800 (sufficient for JSON response)
Response format: JSON object with strict schema

RPC Configuration:
Commitment level: Confirmed (wait for cluster confirmation)
Retry attempts: 3 (with exponential backoff 2^attempt seconds)
Timeout: 60 seconds for HTTP requests

## API Endpoints

The system exposes HTTP endpoints for health monitoring:

GET /
Returns JSON with status healthy, count of agents, count of running agents, and Supabase connection status.

GET /health
Same as root endpoint for load balancer health checks.

GET /api/agents
Returns JSON array of all agents with their complete status including agent_id, address, is_running boolean, loop_count integer, positions_count integer, total_pnl float, win_rate float between 0 and 1, total_trades integer, created_at ISO timestamp, and key_storage string description.

## Project Structure

File layout:
main.py - Core application with all classes and logic
requirements.txt - Python dependencies
README.md - This documentation file
SKILLS.md - Quick reference for AI agents

Main.py contains:
- Import statements and configuration
- DEVNET_POOLS dictionary with verified pool addresses
- ActionType Enum for trade decisions
- Dataclasses: Position, TradeDecision, MarketState, OrcaPool
- SupabaseKeyManager class for encrypted storage
- SolanaClient class for RPC operations
- OrcaWhirlpoolClient class for DEX integration
- AIOracle class for Groq LLM integration
- AgenticWallet class for core trading logic
- MultiAgentSwarm class for multi-agent coordination
- Telegram handler functions
- Web server setup
- Main async entry point

## Dependencies

Required packages with versions:
solana==0.35.0 (Solana Python SDK)
supabase==2.3.4 (Supabase client library)
python-telegram-bot==20.7 (Telegram bot framework)
solders==0.21.0 (Rust-based Solana primitives)
aiohttp==3.9.1 (Async HTTP client)
groq>=0.4.0 (Groq AI client)

Python version: 3.8 or higher required
Operating system: Linux recommended for deployment, works on macOS and Windows for development

## Performance Characteristics

AI decision latency: 50-200ms (Groq inference time)
Transaction confirmation: 2-5 seconds (devnet block time)
Price update frequency: Every 60 seconds (configurable in code)
Memory usage: Approximately 150MB base plus 50MB per active agent
Database storage: Less than 1KB per wallet key

## Troubleshooting Guide

Problem: Bot does not respond to Start button
Solution: Verify TELEGRAM_TOKEN is correct and bot is not blocked. Check logs for connection errors.

Problem: No valid pools loaded error
Solution: Check RPC_URL is accessible. Verify devnet is selected not mainnet. Test RPC endpoint manually.

Problem: AI decision errors or timeouts
Solution: Verify GROQ_API_KEY is valid and has quota remaining. Check internet connectivity. Review Groq status page.

Problem: Keys reset on server restart
Solution: Verify SUPABASE_URL and SUPABASE_KEY are set correctly. Check database table wallet_keys exists. Review logs for Supabase connection errors.

Problem: Transactions fail or simulation errors
Solution: Ensure wallet has devnet SOL balance. Check slippage tolerance is reasonable. Verify token accounts exist.

Problem: Rate limit errors from Groq
Solution: Free tier allows 20 requests per minute. If exceeded, implement delay between requests or upgrade plan.

## Design Philosophy

This system prioritizes security, autonomy, and real execution over simulation. Key principles:
- Real transactions create verifiable on-chain history for judging
- Encryption ensures keys are protected even if database compromised
- Autonomous operation demonstrates true agent capability without human intervention
- Modularity allows swapping components (AI provider, DEX, chain) without rewriting core logic
- Transparency via Telegram gives users visibility into agent decisions and performance

## Legal and Safety Disclaimer

This software executes real transactions on Solana blockchain. While configured for devnet where tokens have no monetary value, the code is capable of mainnet operation if RPC_URL is changed. Users must:
- Never use mainnet RPC with this software unless explicitly adapting for production
- Protect API keys and database credentials as they control real funds
- Understand that AI trading carries risk of loss even on devnet
- Comply with all applicable laws and regulations in their jurisdiction
- Not use for market manipulation or other harmful activities

MIT License applies. Contributors are not liable for losses or damages from use of this software.

## Support and Resources

Documentation:
Solana Python SDK: https://github.com/anza-xyz/solana-py
Orca Whirlpools: https://orca.so/whirlpools
Groq API Reference: https://console.groq.com/docs
Supabase Python: https://supabase.com/docs/reference/python

Communities:
Solana Stack Exchange for technical questions
Superteam Nigeria Discord for bounty-specific questions
Telegram Bot API documentation for bot features

## Development and Contribution

To extend this system:
- Add new DEX integrations by implementing similar client to OrcaWhirlpoolClient
- Swap AI providers by replacing Groq client in AIOracle with OpenAI, Anthropic, or local model
- Add new notification channels by extending Telegram handlers with Discord, email, or SMS
- Implement portfolio rebalancing by extending position management logic
- Add technical indicators by extending MarketState with RSI, MACD, Bollinger Bands calculations

Code style: Follow PEP 8. Use type hints. Document all public methods with docstrings. Keep functions under 50 lines where possible. Use async/await for all I/O operations.

Testing: Test on devnet only. Never commit real private keys. Use environment variables for all secrets. Verify encryption before production deployment.
