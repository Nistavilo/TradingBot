import argparse
import numpy as np
from rich import print
from src.data import make_exchange, fetch_ohlcv_df
from src.strategy.sma import add_sma_signals
from src.broker.paper import PaperBroker
from src.risk import position_size_quote
from src.config import Settings

def run_backtest(symbol: str, timeframe: str, short: int, long: int, limit: int):
    ex = make_exchange(Settings.EXCHANGE, testnet=True)
    df = fetch_ohlcv_df(ex, symbol, timeframe, limit=limit)
    df = add_sma_signals(df, short=short, long=long)

    broker = PaperBroker(init_cash=1000.0, fee_rate=Settings.FEE_RATE)
    equity_curve = []

    position = 0  # 0: flat, 1: long
    for ts, row in df.iterrows():
        price = float(row["close"])
        desired_pos = int(row.get("position", 0))

        # Sinyal uygulama: pozisyon değişimi
        if position != desired_pos:
            if desired_pos == 1 and position <= 0:
                # Long aç: risk kadarıyla al
                quote_amt = position_size_quote(broker.state.cash_quote, Settings.RISK_PER_TRADE)
                broker.buy_market_quote(price, quote_amt)
                position = 1
            elif desired_pos <= 0 and position == 1:
                # Long kapat
                broker.sell_all_market(price)
                position = 0

        equity_curve.append(broker.equity(price))

    if not equity_curve:
        print("[red]Yeterli veri yok.[/red]")
        return

    start_eq = 1000.0
    end_eq = equity_curve[-1]
    ret = (end_eq / start_eq) - 1.0
    max_drawdown = 0.0
    peak = equity_curve[0]
    for x in equity_curve:
        peak = max(peak, x)
        dd = (x / peak) - 1.0
        max_drawdown = min(max_drawdown, dd)

    # Basit metrikler
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252) if len(returns) > 1 else 0.0

    print(f"[bold]Backtest Özeti[/bold]")
    print(f"Sembol: {symbol} | Zaman: {timeframe} | SMA({short},{long}) | Mum: {len(df)}")
    print(f"Başlangıç: {start_eq:.2f} | Bitiş: {end_eq:.2f} | Toplam Getiri: {ret*100:.2f}%")
    print(f"Maks. Gerileme (Drawdown): {max_drawdown*100:.2f}% | Sharpe~: {sharpe:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=Settings.SYMBOL)
    parser.add_argument("--timeframe", default=Settings.TIMEFRAME)
    parser.add_argument("--short", type=int, default=10)
    parser.add_argument("--long", type=int, default=30)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    run_backtest(args.symbol, args.timeframe, args.short, args.long, args.limit)