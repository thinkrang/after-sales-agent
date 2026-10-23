"""评测数据集：所有评测脚本共用这一份。

设计原则
--------
1. 题目必须基于 knowledge/ 和 uploads/ 里的真实文档，不能凭想象编。
2. 分四类，难度递进：
   - easy       直接问，措辞与文档高度重合 → 考基础召回
   - paraphrase 换说法、换口语 → 考语义泛化
   - multihop   需要跨文档或做一步推理 → 考组合能力
   - negative   语义很近但关键信息缺失 → 考抗幻觉（最重要）
   - coreference / slot_fill / switch  多轮三类 → 考上下文（带 history 字段）
3. holdout=True 的题目只用于最终验收，不允许拿来调参、改 prompt。
   一旦你用它调过东西，它就不再是测试题了。

字段说明
--------
question            问题文本
type                题型
expected_sources    期望检索到的文档（用于 Recall / MRR）；空列表表示不参与检索评测
expect_answerable   True=知识库有答案；False=应当拒答
must_include        答案中必须出现的字符串之一（用于判断答对）
expect_refusal      True=应当拒答（negative 题专用）
holdout             是否属于留出集
"""

DATASET = [
    # ==================== easy：直接提问，20 条 ====================
    {"id": "E01", "type": "easy", "question": "商品编号001的单价是多少？",
     "expected_sources": ["product_info.txt"], "expect_answerable": True,
     "must_include": ["59"], "expect_refusal": False, "holdout": False},

    {"id": "E02", "type": "easy", "question": "休闲卫衣多少钱一件？",
     "expected_sources": ["product_info.txt"], "expect_answerable": True,
     "must_include": ["129"], "expect_refusal": False, "holdout": False},

    {"id": "E03", "type": "easy", "question": "运动长裤的库存有多少件？",
     "expected_sources": ["product_info.txt"], "expect_answerable": True,
     "must_include": ["80"], "expect_refusal": False, "holdout": False},

    {"id": "E04", "type": "easy", "question": "纯棉短袖T恤还有多少库存？",
     "expected_sources": ["product_info.txt"], "expect_answerable": True,
     "must_include": ["100"], "expect_refusal": False, "holdout": False},

    {"id": "E05", "type": "easy", "question": "商品签收后几天内可以无理由退货？",
     "expected_sources": ["refund_policy.txt"], "expect_answerable": True,
     "must_include": ["7"], "expect_refusal": False, "holdout": True},

    {"id": "E06", "type": "easy", "question": "退货对商品状态有什么要求？",
     "expected_sources": ["refund_policy.txt"], "expect_answerable": True,
     "must_include": ["完好"], "expect_refusal": False, "holdout": False},

    {"id": "E07", "type": "easy", "question": "超过7天还能无理由退货吗？",
     "expected_sources": ["refund_policy.txt"], "expect_answerable": True,
     "must_include": ["不支持", "不能"], "expect_refusal": False, "holdout": False},

    {"id": "E08", "type": "easy", "question": "商品存在质量问题应该怎么处理？",
     "expected_sources": ["refund_policy.txt"], "expect_answerable": True,
     "must_include": ["售后"], "expect_refusal": False, "holdout": False},

    {"id": "E09", "type": "easy", "question": "付款成功后多久发货？",
     "expected_sources": ["shipping_policy.txt"], "expect_answerable": True,
     "must_include": ["48"], "expect_refusal": False, "holdout": False},

    {"id": "E10", "type": "easy", "question": "默认使用什么快递发货？",
     "expected_sources": ["shipping_policy.txt"], "expect_answerable": True,
     "must_include": ["普通快递"], "expect_refusal": False, "holdout": False},

    {"id": "E11", "type": "easy", "question": "怎么查询订单的物流状态？",
     "expected_sources": ["shipping_policy.txt"], "expect_answerable": True,
     "must_include": ["订单编号"], "expect_refusal": False, "holdout": False},

    {"id": "E12", "type": "easy", "question": "超过48小时还没发货应该怎么办？",
     "expected_sources": ["shipping_policy.txt"], "expect_answerable": True,
     "must_include": ["客服"], "expect_refusal": False, "holdout": False},

    {"id": "E13", "type": "easy", "question": "普通会员每消费1元可以获得多少积分？",
     "expected_sources": ["test_policy.txt"], "expect_answerable": True,
     "must_include": ["1"], "expect_refusal": False, "holdout": True},

    {"id": "E14", "type": "easy", "question": "100积分可以兑换什么？",
     "expected_sources": ["test_policy.txt"], "expect_answerable": True,
     "must_include": ["5元", "优惠券"], "expect_refusal": False, "holdout": False},

    {"id": "E15", "type": "easy", "question": "积分兑换的优惠券有效期是多久？",
     "expected_sources": ["test_policy.txt"], "expect_answerable": True,
     "must_include": ["7天"], "expect_refusal": False, "holdout": False},

    {"id": "E16", "type": "easy", "question": "会员生日当天可以领取什么？",
     "expected_sources": ["birthday_policy.txt"], "expect_answerable": True,
     "must_include": ["10元"], "expect_refusal": False, "holdout": False},

    {"id": "E17", "type": "easy", "question": "生日优惠券的有效期是多久？",
     "expected_sources": ["birthday_policy.txt"], "expect_answerable": True,
     "must_include": ["24小时"], "expect_refusal": False, "holdout": False},

    {"id": "E18", "type": "easy", "question": "生日优惠券每年可以领取几次？",
     "expected_sources": ["birthday_policy.txt"], "expect_answerable": True,
     "must_include": ["一次", "1次"], "expect_refusal": False, "holdout": False},

    {"id": "E19", "type": "easy", "question": "每位员工每年需要完成多少小时的安全培训？",
     "expected_sources": ["employee_policy.md"], "expect_answerable": True,
     "must_include": ["12"], "expect_refusal": False, "holdout": False},

    {"id": "E20", "type": "easy", "question": "安全培训考试多少分算合格？",
     "expected_sources": ["employee_policy.md"], "expect_answerable": True,
     "must_include": ["80"], "expect_refusal": False, "holdout": False},

    # ==================== paraphrase：换说法，15 条 ====================
    {"id": "P01", "type": "paraphrase", "question": "我想买件跑步穿的裤子，价格是多少？",
     "expected_sources": ["product_info.txt"], "expect_answerable": True,
     "must_include": ["89"], "expect_refusal": False, "holdout": False},

    {"id": "P02", "type": "paraphrase", "question": "那件五十九块的上衣还有货吗？",
     "expected_sources": ["product_info.txt"], "expect_answerable": True,
     "must_include": ["100", "有"], "expect_refusal": False, "holdout": False},

    {"id": "P03", "type": "paraphrase", "question": "东西收到八天了，不想要了还能退吗？",
     "expected_sources": ["refund_policy.txt"], "expect_answerable": True,
     "must_include": ["不支持", "不能", "不可以"], "expect_refusal": False, "holdout": True},

    {"id": "P04", "type": "paraphrase", "question": "退回去的商品需要保持什么状态？",
     "expected_sources": ["refund_policy.txt"], "expect_answerable": True,
     "must_include": ["完好", "二次销售"], "expect_refusal": False, "holdout": False},

    {"id": "P05", "type": "paraphrase", "question": "下完单多久能发出来？",
     "expected_sources": ["shipping_policy.txt"], "expect_answerable": True,
     "must_include": ["48"], "expect_refusal": False, "holdout": False},

    {"id": "P06", "type": "paraphrase", "question": "快递单号在哪里可以查到？",
     "expected_sources": ["shipping_policy.txt"], "expect_answerable": True,
     "must_include": ["订单编号"], "expect_refusal": False, "holdout": False},

    {"id": "P07", "type": "paraphrase", "question": "积分能换成钱吗？",
     "expected_sources": ["test_policy.txt"], "expect_answerable": True,
     "must_include": ["优惠券", "5元"], "expect_refusal": False, "holdout": False},

    {"id": "P08", "type": "paraphrase", "question": "兑换来的券多长时间会失效？",
     "expected_sources": ["test_policy.txt"], "expect_answerable": True,
     "must_include": ["7天"], "expect_refusal": False, "holdout": False},

    {"id": "P09", "type": "paraphrase", "question": "我过生日有什么优惠吗？",
     "expected_sources": ["birthday_policy.txt"], "expect_answerable": True,
     "must_include": ["10元", "优惠券"], "expect_refusal": False, "holdout": False},

    {"id": "P10", "type": "paraphrase", "question": "生日礼券能用到什么时候？",
     "expected_sources": ["birthday_policy.txt"], "expect_answerable": True,
     "must_include": ["24小时"], "expect_refusal": False, "holdout": True},

    {"id": "P11", "type": "paraphrase", "question": "公司对员工的信息安全培训有学时要求吗？",
     "expected_sources": ["employee_policy.md"], "expect_answerable": True,
     "must_include": ["12"], "expect_refusal": False, "holdout": False},

    {"id": "P12", "type": "paraphrase", "question": "培训结束之后要考试吗？",
     "expected_sources": ["employee_policy.md"], "expect_answerable": True,
     "must_include": ["考试", "80"], "expect_refusal": False, "holdout": False},

    {"id": "P13", "type": "paraphrase", "question": "那件卫衣卖多少钱？",
     "expected_sources": ["product_info.txt"], "expect_answerable": True,
     "must_include": ["129"], "expect_refusal": False, "holdout": False},

    {"id": "P14", "type": "paraphrase", "question": "长裤还剩多少条？",
     "expected_sources": ["product_info.txt"], "expect_answerable": True,
     "must_include": ["80"], "expect_refusal": False, "holdout": False},

    {"id": "P15", "type": "paraphrase", "question": "在你们店买东西能攒积分吗？怎么算的？",
     "expected_sources": ["test_policy.txt"], "expect_answerable": True,
     "must_include": ["1元", "1积分"], "expect_refusal": False, "holdout": True},

    # ==================== multihop：跨文档或需推理，5 条 ====================
    {"id": "M01", "type": "multihop", "question": "买100元的卫衣能攒多少积分？这些积分能换多少钱的券？",
     "expected_sources": ["test_policy.txt"], "expect_answerable": True,
     "must_include": ["100", "5元"], "expect_refusal": False, "holdout": False},

    {"id": "M02", "type": "multihop", "question": "生日券和积分兑换的券，哪个有效期更长？",
     "expected_sources": ["birthday_policy.txt", "test_policy.txt"], "expect_answerable": True,
     "must_include": ["积分", "7天"], "expect_refusal": False, "holdout": True},

    {"id": "M03", "type": "multihop", "question": "买3件运动长裤一共要多少钱？",
     "expected_sources": ["product_info.txt"], "expect_answerable": True,
     "must_include": ["267"], "expect_refusal": False, "holdout": False},

    {"id": "M04", "type": "multihop", "question": "商品签收第10天发现质量问题，还能申请售后吗？",
     "expected_sources": ["refund_policy.txt"], "expect_answerable": True,
     "must_include": ["可以", "售后"], "expect_refusal": False, "holdout": False},

    {"id": "M05", "type": "multihop", "question": "一件129元的卫衣加一件59元的T恤，总价多少？能攒多少积分？",
     "expected_sources": ["product_info.txt", "test_policy.txt"], "expect_answerable": True,
     "must_include": ["188", "188积分"], "expect_refusal": False, "holdout": False},

    # ==================== negative：近邻难负例，10 条 ====================
    # 共同特征：语义上贴近某个文档，但那个文档里恰好没有这个具体信息。
    {"id": "N01", "type": "negative", "question": "退货运费由谁来承担？",
     "expected_sources": ["refund_policy.txt"], "expect_answerable": False,
     "must_include": ["无法", "没有", "未提及", "不包含"], "expect_refusal": True, "holdout": False},

    {"id": "N02", "type": "negative", "question": "休闲卫衣有XL码吗？",
     "expected_sources": ["product_info.txt"], "expect_answerable": False,
     "must_include": ["无法", "没有", "未提及", "不包含"], "expect_refusal": True, "holdout": True},

    {"id": "N03", "type": "negative", "question": "可以开增值税专用发票吗？",
     "expected_sources": [], "expect_answerable": False,
     "must_include": ["无法", "没有", "未提及", "不包含"], "expect_refusal": True, "holdout": False},

    {"id": "N04", "type": "negative", "question": "积分可以直接兑换商品吗？",
     "expected_sources": ["test_policy.txt"], "expect_answerable": False,
     "must_include": ["无法", "没有", "未提及", "不包含"], "expect_refusal": True, "holdout": True},

    {"id": "N05", "type": "negative", "question": "生日优惠券可以和满减活动叠加使用吗？",
     "expected_sources": ["birthday_policy.txt"], "expect_answerable": False,
     "must_include": ["无法", "没有", "未提及", "不包含"], "expect_refusal": True, "holdout": True},

    {"id": "N06", "type": "negative", "question": "安全培训是在线上进行还是线下集中培训？",
     "expected_sources": ["employee_policy.md"], "expect_answerable": False,
     "must_include": ["无法", "没有", "未提及", "不包含"], "expect_refusal": True, "holdout": False},

    {"id": "N07", "type": "negative", "question": "纯棉短袖T恤有几种颜色可以选？",
     "expected_sources": ["product_info.txt"], "expect_answerable": False,
     "must_include": ["无法", "没有", "未提及", "不包含"], "expect_refusal": True, "holdout": False},

    {"id": "N08", "type": "negative", "question": "会员积分会每年清零吗？",
     "expected_sources": ["test_policy.txt"], "expect_answerable": False,
     "must_include": ["无法", "没有", "未提及", "不包含"], "expect_refusal": True, "holdout": True},

    {"id": "N09", "type": "negative", "question": "发货使用的是哪家快递公司？",
     "expected_sources": ["shipping_policy.txt"], "expect_answerable": False,
     "must_include": ["无法", "没有", "未提及", "不包含"], "expect_refusal": True, "holdout": False},

    {"id": "N10", "type": "negative", "question": "安全培训考试不合格可以补考吗？",
     "expected_sources": ["employee_policy.md"], "expect_answerable": False,
     "must_include": ["无法", "没有", "未提及", "不包含"], "expect_refusal": True, "holdout": True},

    # ==================== multi_turn：多轮对话，12 条 ====================
    # history 是用户前几轮说过的话（按顺序回放，agent 每轮都会真实响应），
    # question 是最后要评测的那一句。
    #
    # 为什么 expected_sources 留空：单轮检索评测用的是 question 单独一句，
    # 而多轮题的"那这个"脱离上下文根本无法检索。混进去会污染 Hit@1 / MRR。
    # 多轮看的是最终答案对不对，单独统计。
    #
    # 三类各 4 条：
    #   coreference 指代消解——"那这个"到底指什么
    #   slot_fill   信息补全——分几轮才把条件问全
    #   switch      话题切换——中途换问题，不能串味

    # ---- 指代消解 ----
    {"id": "T01", "type": "coreference",
     "history": ["我想买件跑步穿的裤子", "运动长裤怎么样？"],
     "question": "那这个多少钱？",
     "expected_sources": [], "expect_answerable": True,
     "must_include": ["89"], "expect_refusal": False, "holdout": False},

    {"id": "T02", "type": "coreference",
     "history": ["商品编号002是什么？"],
     "question": "多少钱一件？",
     "expected_sources": [], "expect_answerable": True,
     "must_include": ["129"], "expect_refusal": False, "holdout": False},

    {"id": "T03", "type": "coreference",
     "history": ["你们退货要几天内？"],
     "question": "那超过7天还能退吗？",
     "expected_sources": [], "expect_answerable": True,
     "must_include": ["不支持", "不能"], "expect_refusal": False, "holdout": False},

    {"id": "T04", "type": "coreference",
     "history": ["会员怎么升级？"],
     "question": "金卡要消费多少？",
     "expected_sources": [], "expect_answerable": True,
     "must_include": ["5000"], "expect_refusal": False, "holdout": False},

    # ---- 信息补全 ----
    {"id": "T05", "type": "slot_fill",
     "history": ["我想查一下库存"],
     "question": "商品编号003的库存还有多少？",
     "expected_sources": [], "expect_answerable": True,
     "must_include": ["80"], "expect_refusal": False, "holdout": False},

    {"id": "T06", "type": "slot_fill",
     "history": ["付款后多久发货？"],
     "question": "如果超过这个时间还没发呢？",
     "expected_sources": [], "expect_answerable": True,
     "must_include": ["客服"], "expect_refusal": False, "holdout": False},

    {"id": "T07", "type": "slot_fill",
     "history": ["订单能改收货地址吗？"],
     "question": "那已经发货了呢？",
     "expected_sources": [], "expect_answerable": True,
     "must_include": ["不可修改", "不能修改", "不能再修改", "无法修改"],
     "expect_refusal": False, "holdout": False},

    {"id": "T08", "type": "slot_fill",
     "history": ["退款多久能到账？"],
     "question": "具体几个工作日？",
     "expected_sources": [], "expect_answerable": True,
     "must_include": ["1至3", "1-3", "1到3"], "expect_refusal": False, "holdout": False},

    # ---- 话题切换 ----
    {"id": "T09", "type": "switch",
     "history": ["退款要几天内？"],
     "question": "对了，你们客服几点上班？",
     "expected_sources": [], "expect_answerable": True,
     "must_include": ["9:00", "9点", "21:00"], "expect_refusal": False, "holdout": False},

    {"id": "T10", "type": "switch",
     "history": ["商品001多少钱？"],
     "question": "换个问题，普通商品多久发货？",
     "expected_sources": [], "expect_answerable": True,
     "must_include": ["48"], "expect_refusal": False, "holdout": False},

    {"id": "T11", "type": "switch",
     "history": ["怎么申请退货？"],
     "question": "那会员有什么权益？",
     "expected_sources": [], "expect_answerable": True,
     "must_include": ["银卡", "金卡", "权益"], "expect_refusal": False, "holdout": False},

    {"id": "T12", "type": "switch",
     "history": ["商品002有货吗？"],
     "question": "顺便问下，修改地址会影响发货吗？",
     "expected_sources": [], "expect_answerable": True,
     "must_include": ["延迟"], "expect_refusal": False, "holdout": False},
]


# ---------------------------------------------------------------- 便捷筛选

def all_cases():
    return list(DATASET)


def retrieval_cases(include_holdout: bool = True):
    """参与检索评测的题目：有期望文档的才纳入。"""
    return [
        c for c in DATASET
        if c["expected_sources"] and (include_holdout or not c["holdout"])
    ]


def answerability_cases(include_holdout: bool = True):
    return [c for c in DATASET if include_holdout or not c["holdout"]]


def agent_cases(include_holdout: bool = True, include_multiturn: bool = False):
    """端到端评测的题目。

    默认**不含**多轮题：多轮的判定口径不同（见 multiturn_cases），
    混在一起会让"任务成功率"这个数字失去可比性。
    """
    return [
        c for c in DATASET
        if (include_holdout or not c["holdout"])
        and (include_multiturn or not c.get("history"))
    ]


def multiturn_cases(include_holdout: bool = True):
    """多轮题：靠 history 回放上下文，单独统计。"""
    return [
        c for c in DATASET
        if c.get("history") and (include_holdout or not c["holdout"])
    ]


def stats():
    from collections import Counter
    c = Counter(x["type"] for x in DATASET)
    h = sum(1 for x in DATASET if x["holdout"])
    return dict(total=len(DATASET), by_type=dict(c), holdout=h)


if __name__ == "__main__":
    s = stats()
    print(f"题目总数: {s['total']}")
    for k, v in s["by_type"].items():
        print(f"  {k:<12} {v} 条")
    print(f"留出集(holdout): {s['holdout']} 条")
    print(f"参与检索评测: {len(retrieval_cases())} 条")
