import time
from typing import Optional, List


def listr(lst: Optional[List[List]] = None) -> str:
    if lst is None or not len(lst):
        return ""

    len_list = [0] * len(lst[0])
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            len_list[j] = max(len_list[j], len(str(lst[i][j])))

    res = ""
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            res += f"{str(lst[i][j]).rjust(len_list[j])} "
        res += "\n"
    return res


class MyTimer(object):
    TITLE = ["id", "name", "time", "per"]

    def __init__(self, name: Optional[str] = None):
        self.name = name if name is not None else f"{self.now()}"
        self.timer: Optional[float] = None
        self.result_list: List[List] = list()

        self.start()

    def now(self):
        return time.time()

    def start(self):
        self.timer = self.now()

    def stop(self, name: Optional[str] = None):
        if self.timer is None:
            raise RuntimeError("You should start timer.")

        if name is None:
            name = f"seg-{len(self.result_list)}"

        result_dict = {
            "id": len(self.result_list),
            "name": name,
            "time": round(self.now() - self.timer, 2),
            "per": 0.
        }
        self.result_list.append([result_dict[k] for k in MyTimer.TITLE])
        self.timer = None
        self._calc_per()

    def _calc_per(self):
        total_time = sum([x[2] for x in self.result_list])
        for x in self.result_list:
            if total_time < 1e-9:
                x[3] = round(x[2] / len(self.result_list), 4) * 100
            else:
                x[3] = round(x[2] / total_time, 4) * 100

    def lap(self, name: Optional[str] = None):
        self.stop(name)
        self.start()

    def lap_and_print(self, name: Optional[str] = None, logger=None, print_all: bool = False):
        self.lap(name)
        repr_str = " ".join([str(x) for x in self.result_list[-1]]) if len(self.result_list) > 0 else ""
        if print_all:
            repr_str = self.__repr__()
        if logger is None:
            print(repr_str)
        else:
            logger.info(repr_str)
        return repr_str

    def __repr__(self):
        temp = [MyTimer.TITLE]
        temp.extend(self.result_list)
        return f"Timer-{self.name}\n{listr(temp)}"


if __name__ == "__main__":
    my_timer = MyTimer()
    my_timer.start()
    time.sleep(1)
    my_timer.lap_and_print()
    time.sleep(2)
    my_timer.lap_and_print()
    time.sleep(3)
    my_timer.lap_and_print()
