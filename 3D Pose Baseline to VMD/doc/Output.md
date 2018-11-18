# 出力ファイルの詳細

## smoothed.txt

- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) の解析結果を滑らかにした2D関節データ
- 1行が1フレームに相当しており、全フレームの情報が1ファイルに集約されています。

### OpenpseのOutputフォーマット
![OpenpseのOutputフォーマット](openpose_body.png)

> 引用元：[OpenPose - Output](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md)

### smoothed.txt の 1フレーム(一行)内のデータ構造

0. 鼻 - X位置
1. 鼻 - Y位置
2. 首 - X位置
3. 首 - Y位置
4. 右肩 - X位置
5. 右肩 - Y位置
6. 右ひじ - X位置
7. 右ひじ - Y位置
8. 右手首 - X位置
9. 右手首 - Y位置
10. 左肩 - X位置
11. 左肩 - Y位置
12. 左ひじ - X位置
13. 左ひじ - Y位置
14. 左手首 - X位置
15. 左手首 - Y位置
16. 右尻(右足付け根) - X位置
17. 右尻(右足付け根) - X位置
18. 右ひざ - X位置
19. 右ひざ - Y位置
20. 右足首 - X位置
21. 右足首 - Y位置
22. 左尻(左足付け根) - X位置
23. 左尻(左足付け根) - X位置
24. 左ひざ - X位置
25. 左ひざ - Y位置
26. 左足首 - X位置
27. 左足首 - Y位置
28. 右目 - X位置
29. 右目 - Y位置
30. 左目 - X位置
31. 左目 - Y位置
32. 右耳 - X位置
33. 右耳 - Y位置
34. 左耳 - X位置
35. 左耳 - Y位置

## pos.txt

- [3d-pose-baseline-vmd](https://github.com/miu200521358/3d-pose-baseline-vmd) の実行結果として得られた3D関節データ
- 1行が1フレームに相当しており、全フレームの情報が1ファイルに集約されています。

### 3d-pose-baseline のOutputフォーマット

![3d-pose-baseline のOutputフォーマット](3d-pose-baseline.png)

### smoothed.txt の 1フレーム(一行)内のデータ構造

1関節ごとにカンマで区切り、その中で、X, Y, Z 軸が空白区切りとなっています。

例）... 右ひざx 右ひざy 右ひざz, 右足首x 右足首y 右足首z, ...

0. 尻(尾てい骨)
1. 右尻(右足付け根)
2. 右ひざ
3. 右足首
4. 左尻(左足付け根)
5. 左ひざ
6. 左足首
7. 脊椎
8. 胸
9. 首/鼻
10. 頭
11. 左肩
12. 左ひじ
13. 左手首
14. 右肩
15. 右ひじ
16. 右手首
