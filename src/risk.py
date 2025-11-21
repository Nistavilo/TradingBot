def position_size_quote(balance_quote: float, risk_per_trade: float) -> float:
    """
    Bakiye bazlı sabit fraksiyonel model:
    - Her işlemde, mevcut QUOTE (ör. USDT) bakiyenin risk_per_trade kadarı kullanılır.
    Not: Stop yoksa 'risk' kelimesi nominaldir; geliştirmede ATR/stop tabanlı boyutlama önerilir.
    """
    return max(0.0, balance_quote * risk_per_trade)