# -*- coding: utf-8 -*-
"""메이크업 경로 후처리 — 학습 코드에서 옮긴 규칙이 그대로인지 지킨다.

여기 값들(랜드마크 번호·임계값·문장 표현)은 학습 정답을 만들 때 쓴 것과
같아야 한다. 다르면 좌표가 맞아도 엉뚱한 부위를 말하게 되고, 사용자는 화면을
못 보므로 틀렸다는 걸 알 방법이 없다.
"""
import pytest

from beautytalk.makeup_regions import (
    CLOSING,
    FINE_HIST,
    FINE_LABEL_MIN_COUNT,
    NO_DEFECT,
    answer_from_boxes,
    coarsen_fine,
    crop_norm_to_bbox,
    describe_region_band,
    is_wide_area,
    region_label,
    parse_coord_answer,
    state_and_action,
)


def landmarks():
    """정면 얼굴 하나. describe_region_band 가 참조하는 점만 채운다."""
    pts = [(0.0, 0.0)] * 470
    pts[10] = (500, 100)                       # 이마 맨 위
    for i in (105, 334, 107, 336):
        pts[i] = (500, 300)                    # 눈썹 선
    pts[145] = (430, 400); pts[374] = (570, 400)   # 아래 눈꺼풀
    pts[2] = (500, 520)                        # 코 밑
    pts[17] = (500, 660)                       # 아랫입술 아래
    for i in (6, 168, 4, 1):
        pts[i] = (500, 450)                    # 코 중심선
    pts[102] = (450, 500); pts[331] = (550, 500)   # 콧방울 (반폭 50)
    pts[61] = (430, 600); pts[291] = (570, 600)    # 입 (반폭 70)
    return pts


LM = landmarks()


def box(cx, cy, s=10):
    return [cx - s, cy - s, cx + s, cy + s]


class Test파싱:
    def test_정상_배열(self):
        r = parse_coord_answer('[{"bbox_2d": [10, 20, 30, 40], "label": "경계"}]')
        assert r == [([10.0, 20.0, 30.0, 40.0], "boundary")]

    def test_빈_배열(self):
        assert parse_coord_answer("[]") == []

    def test_잘린_출력도_건져낸다(self):
        """토큰 제한에 걸려 배열이 안 닫혀도 완성된 객체까지는 쓴다."""
        r = parse_coord_answer('[{"bbox_2d":[1,2,3,4],"label":"덜발림"},{"bbox_2d":[5,6')
        assert r == [([1.0, 2.0, 3.0, 4.0], "missing")]

    def test_뒤집힌_박스는_버린다(self):
        assert parse_coord_answer('[{"bbox_2d":[30,40,10,20],"label":"경계"}]') == []

    def test_모르는_라벨은_불균형(self):
        r = parse_coord_answer('[{"bbox_2d":[1,2,3,4],"label":"이상함"}]')
        assert r[0][1] == "uneven"

    def test_두_개까지만(self):
        boxes = ",".join(f'{{"bbox_2d":[{i},{i},{i+5},{i+5}],"label":"경계"}}' for i in range(1, 6))
        assert len(parse_coord_answer(f"[{boxes}]")) == 2


class Test좌표변환:
    def test_크롭_정규화를_원본_좌표로(self):
        # rect=(100,200,400,600) 안에서 0~1000 → 원본 픽셀
        assert crop_norm_to_bbox([0, 0, 1000, 1000], (100, 200, 400, 600)) == [100, 200, 500, 800]
        assert crop_norm_to_bbox([500, 500, 500, 500], (0, 0, 1000, 1000)) == [500, 500, 500, 500]


class Test부위:
    @pytest.mark.parametrize("cx,cy,expect", [
        (500, 150, "이마 중앙"),
        (500, 280, "미간"),
        (500, 350, "콧대"),
        (500, 500, "코끝"),
        (500, 550, "인중"),
        (500, 700, "턱 중앙"),
    ])
    def test_중앙_세로_구간(self, cx, cy, expect):
        assert describe_region_band(box(cx, cy), LM) == expect

    def test_이미지_왼쪽은_사용자_오른쪽(self):
        """코 중심선 기준. 여기서 뒤집히면 사용자가 반대쪽을 문지른다."""
        assert describe_region_band(box(200, 350), LM).endswith("오른쪽")
        assert describe_region_band(box(800, 350), LM).endswith("왼쪽")

    def test_콧방울_옆은_콧볼(self):
        # 콧방울 반폭 50 → 1.8배(90) 안쪽
        assert describe_region_band(box(430, 470), LM) == "콧볼 오른쪽"

    def test_바깥쪽은_볼(self):
        assert describe_region_band(box(200, 500), LM) == "볼 위쪽 오른쪽"

    def test_랜드마크_없으면_None(self):
        assert describe_region_band(box(500, 500), None) is None


class Test상위표현:
    @pytest.mark.parametrize("fine,head", [
        ("눈두덩이 왼쪽", "눈 왼쪽"),
        ("볼 아래쪽 오른쪽", "볼 오른쪽"),
        ("콧볼 왼쪽", "코 왼쪽"),
        ("이마 중앙", "이마"),
        ("턱 중앙", "턱"),
    ])
    def test_한_단계만_낮춘다(self, fine, head):
        assert coarsen_fine(fine) == head

    def test_모르는_이름은_None(self):
        assert coarsen_fine("귓불 왼쪽") is None


class Test히스토그램:
    """학습 데이터에 충분히 나온 라벨만 세밀하게 부른다."""

    def test_히스토그램이_실려있다(self):
        assert FINE_HIST, "fine_hist.json 을 못 읽으면 전부 상위 표현으로 뭉개진다"
        assert sum(FINE_HIST.values()) == 873

    def test_예시가_많으면_세밀하게(self):
        # 볼 아래쪽 오른쪽: 63회 >= 15 -> 세밀 라벨 유지
        assert FINE_HIST["볼 아래쪽 오른쪽"] >= FINE_LABEL_MIN_COUNT
        assert region_label(box(200, 600), LM) == "볼 아래쪽 오른쪽"
        # 턱 오른쪽: 28회 -> 역시 유지
        assert region_label(box(200, 700), LM) == "턱 오른쪽"

    def test_예시가_적으면_상위_표현으로(self):
        # 눈두덩이 왼쪽: 14회 < 15 -> 눈 왼쪽
        assert FINE_HIST["눈두덩이 왼쪽"] < FINE_LABEL_MIN_COUNT
        assert region_label(box(800, 350), LM) == "눈 왼쪽"
        # 눈두덩이 오른쪽: 21회 >= 15 -> 그대로
        assert FINE_HIST["눈두덩이 오른쪽"] >= FINE_LABEL_MIN_COUNT
        assert region_label(box(200, 350), LM) == "눈두덩이 오른쪽"

    def test_랜드마크_없으면_None(self):
        assert region_label(box(500, 500), None) is None


class Test넓은범위:
    def test_얼굴_28퍼센트_넘으면_넓은_범위(self):
        face = (0, 0, 100, 100)
        assert is_wide_area([0, 0, 30, 10], face) is True
        assert is_wide_area([0, 0, 10, 10], face) is False

    def test_얼굴박스_없으면_False(self):
        assert is_wide_area([0, 0, 999, 999], None) is False


class Test문장:
    def test_받침에_따라_조사가_바뀐다(self):
        """턱이 / 코가 — 여기가 틀리면 읽을 때 바로 어색하다."""
        assert state_and_action("턱", False, "missing")[0].startswith("턱이")
        assert state_and_action("코", False, "missing")[0].startswith("코가")

    def test_종류별로_동작이_다르다(self):
        """덜 발린 곳을 두드려봤자 안 채워진다."""
        assert "펴 발라" in state_and_action("이마", False, "missing")[1]
        assert "문질러" in state_and_action("이마", False, "boundary")[1]
        assert "두드려" in state_and_action("이마", False, "uneven")[1]

    def test_결함_없으면_안심시킨다(self):
        msg, detail, items = answer_from_boxes([], LM, None, (0, 0, 1000, 1000))
        assert msg == NO_DEFECT and detail == "none"

    def test_한_곳_안내(self):
        pred = [([180, 330, 220, 370], "boundary")]
        msg, detail, items = answer_from_boxes(pred, LM, (400, 200, 600, 700), (0, 0, 1000, 1000))
        assert "오른쪽" in msg and CLOSING in msg
        assert detail.startswith("1곳")

    def test_좌표는_절대_읽지_않는다(self):
        pred = [([180, 330, 220, 370], "missing")]
        msg, _, _ = answer_from_boxes(pred, LM, None, (0, 0, 1000, 1000))
        for n in ("180", "330", "220", "370"):
            assert n not in msg

    def test_같은_부위_같은_종류는_한_번만(self):
        """같은 문장을 두 번 말하는 사례가 실제로 있었다."""
        pred = [([180, 330, 220, 370], "boundary"), ([185, 335, 225, 375], "boundary")]
        _, detail, _ = answer_from_boxes(pred, LM, None, (0, 0, 1000, 1000))
        assert detail.startswith("1곳")

    def test_화면용_항목이_문장과_함께_나온다(self):
        """화면은 부위/상태/동작을 따로 그린다. 문장을 다시 쪼개면 어긋나므로
        만들 때 같이 내보낸다."""
        pred = [([180, 330, 220, 370], "boundary")]
        msg, _, items = answer_from_boxes(pred, LM, None, (0, 0, 1000, 1000))
        assert len(items) == 1
        it = items[0]
        assert set(it) == {"region", "state", "action", "type"}
        assert it["type"] == "경계"
        assert it["region"] in msg          # 화면과 음성이 같은 부위를 가리킨다
        assert it["state"] in msg
        assert it["action"] in msg

    def test_결함_없으면_항목도_비어_있다(self):
        _, _, items = answer_from_boxes([], LM, None, (0, 0, 1000, 1000))
        assert items == []

    def test_얼굴_밖_예측은_버린다(self):
        """화장 도구에 묻은 제품을 결함으로 잡는 사례가 있었다."""
        pred = [([180, 330, 220, 370], "boundary")]
        msg, detail, items = answer_from_boxes(pred, LM, None, (0, 0, 1000, 1000),
                                        on_face=lambda b: False)
        assert msg == NO_DEFECT and detail == "off_face"
