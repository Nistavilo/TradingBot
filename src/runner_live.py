import argparse
import time
from rich import print
from src.data import make_exchange, fetch_ohlcv_df, append_latest_candle
from src.strategy.sma import add_sma_signals
from src.broker.paper import PaperBroker
from src.risk import position_size_quote
from src.config import Settings

def run_live(symbol: str, timeframe: str, short: int, long: int):
    ex = make_exchange(Settings.EXCHANGE, Settings.API_KEY, Settings.API_SECRET, Settings.TESTNET)

    # Başlangıç verisi
    df = fetch_ohlcv_df(ex, symbol, timeframe, limit=200)
    df = add_sma_signals(df, short=short, long=long)

    if Settings.PAPER_TRADING:
        broker = PaperBroker(init_cash=1000.0, fee_rate=Settings.FEE_RATE)
        print("[yellow]Paper trading modunda çalışıyor...[/yellow]")
    else:
        broker = None
        print("[green]Canlı mod: Gerçek emir gönderecek![/green]")

    last_seen_ts = df.index[-1] if len(df) else None
    position = 0

    print(f"Sembol: {symbol} | Zaman: {timeframe} | SMA({short},{long})")
    while True:
        try:
            # Veriyi güncelle
            df = append_latest_candle(df, ex, symbol, timeframe)
            df = add_sma_signals(df, short=short, long=long)

            current_ts = df.index[-1]
            row = df.iloc[-1]
            price = float(row["close"])
            desired_pos = int(row.get("position", 0))

            # Yeni mum kapandı mı? Basit kontrol: timestamp değiştiyse
            new_candle = (last_seen_ts is None) or (current_ts != last_seen_ts)
            if new_candle:
                # Yalnızca yeni mumda aksiyon
                if position != desired_pos:
                    if desired_pos == 1 and position <= 0:
                        if Settings.PAPER_TRADING:
                            quote_amt = position_size_quote(broker.state.cash_quote, Settings.RISK_PER_TRADE)
                            broker.buy_market_quote(price, quote_amt)
                            print(f"[cyan]{current_ts}[/cyan] LONG açıldı @ {price:.2f}, Equity: {broker.equity(price):.2f}")
                        else:
                            # Gerçek emir (spot market al)
                            amt_quote = 100.0  # Örnek sabit miktar: geliştirin
                            qty = (amt_quote / price)
                            order = ex.create_order(symbol, type="market", side="buy", amount=qty)
                            print(f"[green]{current_ts} LONG order gönderildi[/green]: {order.get('id')}")
                        position = 1
                    elif desired_pos <= 0 and position == 1:
                        if Settings.PAPER_TRADING:
                            broker.sell_all_market(price)
                            print(f"[cyan]{current_ts}[/cyan] LONG kapatıldı @ {price:.2f}, Equity: {broker.equity(price):.2f}")
                        else:
                            # Gerçek emir (spot market sat) - dikkat: eldeki miktarı öğren
                            bal = ex.fetch_balance()
                            base_ccy = symbol.split("/")[0]
                            qty = bal.get(base_ccy, {}).get("free", 0.0)
                            if qty and qty > 0:
                                order = ex.create_order(symbol, type="market", side="sell", amount=qty)
                                print(f"[green]{current_ts} FLAT order gönderildi[/green]: {order.get('id')}")
                        position = 0

                last_seen_ts = current_ts

            # Polling aralığı (timeframe'e göre ayarlayabilirsiniz)
            time.sleep(10)

        except KeyboardInterrupt:
            print("[red]Durduruldu.[/red]")
            break
        except Exception as e:
            print(f"[red]Hata:[/red] {e}")
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=Settings.SYMBOL)
    parser.add_argument("--timeframe", default=Settings.TIMEFRAME)
    parser.add_argument("--short", type=int, default=10)
    parser.add_argument("--long", type=int, default=30)
    args = parser.parse_args()
    run_live(args.symbol, args.timeframe, args.short, args.long)