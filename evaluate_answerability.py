
import time
from test_answerability import check_answerability

# 知识库有答案的问题
POSITIVE = [
    "商品签收八天，不喜欢了还能退货吗？",
    "七天无理由退货有什么条件？",
    "商品存在质量问题应该怎么处理？",
    "付款后多久可以发货？",
    "怎么查询订单物流状态？",
    "48小时还没发货怎么办？",
    "编号001的商品多少钱？",
    "运动长裤的价格是多少？",
    "休闲卫衣的单价是多少？",
]

# 知识库没有明确答案的问题
NEGATIVE = [
    "店铺支持配送到火星吗？",
    "店铺可以使用比特币付款吗？",
    "店铺在北京有多少家实体门店？",
    "员工的基本月薪是多少？",
    "商品提供终身免费维修吗？",
    "店铺客服电话是多少？",
    "明年双十一全场打几折？",
    "店铺的创始人是谁？",
]


def evaluate():

    cases = (
        [(q, "YES") for q in POSITIVE]
        + [(q, "NO") for q in NEGATIVE]
    )

    tp = tn = fp = fn = errors = 0
    total_time = 0

    for i, (question, expected) in enumerate(cases, 1):

        start = time.perf_counter()

        try:
            result = check_answerability(question)
            actual = result.split("|", 1)[0].strip().upper()
        except Exception as e:
            result = f"ERROR|{e}"
            actual = "ERROR"

        elapsed = time.perf_counter() - start
        total_time += elapsed

        if actual not in ("YES", "NO"):
            errors += 1
        elif expected == "YES" and actual == "YES":
            tp += 1
        elif expected == "YES" and actual == "NO":
            fn += 1
        elif expected == "NO" and actual == "NO":
            tn += 1
        else:
            fp += 1

        status = "PASS" if actual == expected else "FAIL"

        print(f"\n[{i}/{len(cases)}] {question}")
        print(f"预期: {expected} | 实际: {actual}")
        print(f"结果: {status}")
        print(f"模型输出: {result}")
        print(f"耗时: {elapsed:.2f}s")

    print("\n========== Answerability Evaluation ==========")

    positive_total = len(POSITIVE)
    negative_total = len(NEGATIVE)
    total = len(cases)

    print(f"总问题数: {total}")
    print(f"正确接受 TP: {tp}")
    print(f"正确拒答 TN: {tn}")
    print(f"错误接受 FP: {fp}")
    print(f"错误拒答 FN: {fn}")
    print(f"异常或无效输出: {errors}")

    print(f"\n正例接受率: {tp / positive_total:.2%}")
    print(f"负例拒答率: {tn / negative_total:.2%}")
    print(f"错误接受率: {fp / negative_total:.2%}")
    print(f"错误拒答率: {fn / positive_total:.2%}")
    print(f"总体正确率: {(tp + tn) / total:.2%}")
    print(f"平均核查耗时: {total_time / total:.2f}s")


if __name__ == "__main__":
    evaluate()