from dataclasses import dataclass

@dataclass
class PaperState:
    cash_quote: float = 1000.0   # örn. USDT
    asset_qty: float = 0.0       # örn. BTC miktarı
    fee_rate: float = 0.0006

class PaperBroker:
    def __init__(self, init_cash: float = 1000.0, fee_rate: float = 0.0006):
        self.state = PaperState(cash_quote=init_cash, fee_rate=fee_rate)

    def equity(self, price: float) -> float:
        return self.state.cash_quote + self.state.asset_qty * price

    def buy_market_quote(self, price: float, quote_amount: float):
        if quote_amount <= 0: 
            return
        quote_amount = min(quote_amount, self.state.cash_quote)
        fee = quote_amount * self.state.fee_rate
        net = max(0.0, quote_amount - fee)
        qty = net / price
        self.state.cash_quote -= quote_amount
        self.state.asset_qty += qty

    def sell_all_market(self, price: float):
        qty = self.state.asset_qty
        if qty <= 0:
            return
        gross = qty * price
        fee = gross * self.state.fee_rate
        net = max(0.0, gross - fee)
        self.state.asset_qty = 0.0
        self.state.cash_quote += net