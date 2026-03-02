from datetime import datetime
from typing import List

def generate_month_range(start_month: str, end_month: str) -> List[str]:
    """
    生成两个月份之间的所有月份序列（包含起止月份）。

    Args:
        start_month: 起始月份，格式 'yyyy-mm'
        end_month: 结束月份，格式 'yyyy-mm'

    Returns:
        月份字符串列表，如 ['2022-01', '2022-02', ..., '2022-12']

    Example:
        generate_month_range('2022-03', '2022-06')
        # 返回 ['2022-03', '2022-04', '2022-05', '2022-06']
    """
    start = datetime.strptime(start_month, '%Y-%m')
    end = datetime.strptime(end_month, '%Y-%m')

    months = []
    current = start
    while current <= end:
        months.append(current.strftime('%Y-%m'))
        # 下一个月
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    return months