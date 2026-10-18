import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import dryrun_check as D
D.WORK.mkdir(parents=True, exist_ok=True)
print("### 正式种子（修复后）")
n1, ok1, w1 = D.check_frozen(legacy=False)
print(f"\n>>> {n1}: {'PASS' if ok1 else 'FAIL'}  {w1}")
print("\n\n### 旧常数种子（被修掉的行为）")
n2, ok2, w2 = D.check_frozen(legacy=True)
print(f"\n>>> {n2}: {'PASS' if ok2 else 'FAIL'}  {w2}")
print(f"\n结论: 修复后 {'通过' if ok1 else '不通过'}; 旧行为 {'锁死（检查有牙）' if not ok2 else '居然也通过 —— 检查没牙'}")
