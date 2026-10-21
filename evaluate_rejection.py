
from rag import debug_retrieval

# 已知知识库中有答案的问题
POSITIVE_QUESTIONS = [
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

# 人工确认当前知识库没有明确答案的问题
NEGATIVE_QUESTIONS = [
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
    positive_distances = []
    negative_distances = []

    print("\n========== 有答案问题 ==========")

    for question in POSITIVE_QUESTIONS:
        results = debug_retrieval(question, top_k=3)

        if not results:
            print("无检索结果：", question)
            continue

        first = results[0]
        distance = first["distance"]
        positive_distances.append(distance)

        print(f"\n问题：{question}")
        print(f"Top1：{first['source']}")
        print(f"Distance：{distance:.4f}")

    print("\n========== 无答案问题 ==========")

    for question in NEGATIVE_QUESTIONS:
        results = debug_retrieval(question, top_k=3)

        if not results:
            print("无检索结果：", question)
            continue

        first = results[0]
        distance = first["distance"]
        negative_distances.append(distance)

        print(f"\n问题：{question}")
        print(f"Top1：{first['source']}")
        print(f"Distance：{distance:.4f}")

    print("\n========== 距离分布分析 ==========")

    if positive_distances and negative_distances:
        positive_max = max(positive_distances)
        negative_min = min(negative_distances)

        print(f"正例 Top1 最大距离：{positive_max:.4f}")
        print(f"负例 Top1 最小距离：{negative_min:.4f}")

        if positive_max < negative_min:
            print("当前样本的距离存在可分离区间。")
            print("但仍需要独立测试集验证阈值。")
        else:
            print("正负样本距离出现重叠。")
            print("单一距离阈值无法完美区分当前样本。")


if __name__ == "__main__":
    evaluate()