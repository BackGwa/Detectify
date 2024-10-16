from .logger import Logger


log = Logger()


def ask(message: str) -> bool:
    while True:
        select = input(f"{message} (Y/n) : ").lower()
        if select == 'y' or select == '':
            return True
        elif select == 'n':
            return False
        else:
            log.warn("입력 값은 Y/n 이어야합니다.")