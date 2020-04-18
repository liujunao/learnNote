参考：[https://blog.csdn.net/jmh1996/article/details/78289915](https://blog.csdn.net/jmh1996/article/details/78289915) 

## 1、矩阵与行列式

- **矩阵**：
  $$
  D(q) =
  \begin{bmatrix}
  p_1+p_2+2p_3cosq_{2} & p_2+p_3cosq_2\\
  p_2+p_3cosq_2 & p_2 
  \end{bmatrix}
  $$

  ```
  $$
  D(q) = \begin{bmatrix}
  			p_1+p_2+2p_3cosq_{2} & p_2+p_3cosq_2
  			\\
  			p_2+p_3cosq_2 & p_2 
  		\end{bmatrix}
  $$
  ```

- 省略号：
  $$
  \begin{pmatrix}
  a_{11} & \dots & a_{1n}\\
  \vdots & \ddots & \vdots\\
  a_{m1} & \dots & a_{mn}
  \end{pmatrix}
  $$

  ```
  $$
  \begin{pmatrix}
  	a_{11} & \dots & a_{1n}  \\
  	\vdots & \ddots & \vdots \\
  	a_{m1} & \dots & a_{mn}
  \end{pmatrix}
  $$
  ```

- 行列式：
  $$
  \begin{vmatrix}
  1 & 0 \\
  0 & -1 \end{vmatrix}
  $$

  ```
  $$
  \begin{vmatrix}
  	1 & 0 \\
  	0 & -1 
  \end{vmatrix}
  $$
  ```

  $$
  \begin{Vmatrix}
  1 & 0 \\
  0 & -1 \end{Vmatrix}
  $$

  ```
  $$
  \begin{Vmatrix}
  	1 & 0 \\
  	0 & -1 
  \end{Vmatrix}
  $$
  ```

## 2、方程组

$$
\begin{cases} 方程式一\\ 方程式二\\ 方程式三\\ \end{cases}
$$

```
$$
\begin{cases} 
	方程式一\\ 
	方程式二\\ 
	方程式三\\ 
\end{cases}
$$
```

## 3、希腊字母

| 序号 |    大写    |     小写      |   md 语法   | 中文名称 |
| :--: | :--------: | :-----------: | :---------: | :------: |
|  1   |  $\Alpha$  |   $\alpha$    |   \alpha    |  阿尔法  |
|  2   |  $\Beta$   |    $\beta$    |    \beta    |   贝塔   |
|  3   |  $\Gamma$  |   $\gamma$    |   \gamma    |   伽马   |
|  4   |  $\Delta$  |   $\delta$    |   \delta    |  德尔塔  |
|  5   | $\Epsilon$ |  $\epsilon$   |  \epsilon   | 伊普西隆 |
|  6   |  $\Zeta$   |    $\zeta$    |    \zeta    |   泽塔   |
|  7   |   $\Eta$   |    $\eta$     |    \eta     |   伊塔   |
|  8   |  $\Theta$  |   $\theta$    |   \theta    |   西塔   |
|  9   |  $\Iota$   |    $\iota$    |    \iota    |   约塔   |
|  10  |  $\Kappa$  |   $\kappa$    |   \kappa    |   卡帕   |
|  11  | $\Lambda$  |   $\lambda$   |   \lambda   |  兰姆达  |
|  12  |   $\Mu$    |     $\mu$     |     \mu     |    缪    |
|  13  |   $\Nu$    |     $\nu$     |     \nu     |    纽    |
|  14  |   $\Xi$    |     $\xi$     |     \xi     |   克西   |
|  15  | $\Omicron$ |  $\omicron$   |  \omicron   | 欧米克隆 |
|  16  |   $\Pi$    |     $\pi$     |     \pi     |    派    |
|  17  |   $\Rho$   |    $\rho$     |    \rho     |    柔    |
|  18  |  $\Sigma$  |   $\sigma$    |   \sigma​    |  西格玛  |
|  19  |   $\Tau$   |    $\tau$     |    \tau     |    陶    |
|  20  | $\Upsilon$ |  $\upsilon$   |  \upsilon   | 宇普西隆 |
|  21  |   $\Phi$   |    $\phi$     |    \phi     |   弗爱   |
|  22  |   $\Chi$   |    $\chi$     |    \chi     |    卡    |
|  23  |   $\Psi$   |    $\psi$     |    \psi     |   普赛   |
|  24  |  $\Omega$  |   $\omega$    |   \omega    |  欧米伽  |
|      |            | $\varepsilon$ | \varepsilon |          |
|      |            |  $\varkappa$  |  \varkappa  |          |
|      |            |  $\vartheta$  |  \vartheta  |          |
|      |            |   $\varpi$    |   \varpi    |          |
|      |            |   $\varrho$   |   \varrho   |          |
|      |            |  $\varsigma$  |  \varsigma  |          |
|      |            |   $\varphi$   |   \varphi   |          |

## 4、二元关系符

> 命令前添加 `\not` 构成否定，如： `\not\leq -->` $\not\leq$ 

|   $\leq$    |   $\geq$    |   $\equiv$    |     $\ll$     |   $\gg$   |  $\doteq$   |  $\sim$  | $\simeq$  |
| :---------: | :---------: | :-----------: | :-----------: | :-------: | :---------: | :------: | :-------: |
|    \leq     |    \geq     |    \equiv     |      \ll      |    \gg    |   \doteq    |   \sim   |  \simeq   |
|   $\prec$   |   $\succ$   |   $\preceq$   |   $\succeq$   | $\approx$ |   $\cong$   | $\Join$  | $\bowtie$ |
|    \prec    |    \succ    |    \preceq    |    \succeq    |  \approx  |    \cong    |  \Join   |  \bowtie  |
|  $\subset$  |  $\supset$  |  $\subseteq$  |  $\supseteq$  |   $\in$   |    $\ni$    | $\notin$ | $\propto$ |
|   \subset   |   \supset   |   \subseteq   |   \supseteq   |    \in    |     \ni     |  \notin  |  \propto  |
| $\sqsubset$ | $\sqsupset$ | $\sqsubseteq$ | $\sqsupseteq$ |  $\neq$   |  $\smile$   | $\frown$ | $\asymp$  |
|  \sqsubset  |  \sqsupset  |  \sqsubseteq  |  \sqsupseteq  |   \neq    |   \smile    |  \frown  |  \asymp   |
|  $\vdash$   |  $\dashv$   |   $\models$   |    $\mid$     |  $\nmid$  | $\parallel$ | $\perp$  |           |
|   \vdash    |   \dashv    |    \models    |     \mid      |   \nmid   |  \parallel  |  \perp   |           |

## 5、 二元运算符

|  $\pm$   |  $\mp$  |  $\cdot$  |  $\div$   |  $\times$  | $\setminus$ |      $\amalg$      |
| :------: | :-----: | :-------: | :-------: | :--------: | :---------: | :----------------: |
|   \pm    |   \mp   |   \cdot   |   \div    |   \times   |  \setminus  |       \amalg       |
| $\star$  | $\ast$  |  $\circ$  | $\bullet$ | $\diamond$ |   $\lhd$    |  $\bigtriangleup$  |
|  \star   |  \ast   |   \circ   |  \bullet  |  \diamond  |    \lhd     |   \bigtriangleup   |
| $\oplus$ | $\odot$ | $\otimes$ | $\ominus$ | $\oslash$  |   $\rhd$    | $\bigtriangledown$ |
|  \oplus  |  \odot  |  \otimes  |  \ominus  |  \oslash   |    \rhd     |  \bigtriangledown  |
|  $\cup$  | $\cap$  | $\sqcup$  | $\sqcap$  |   $\vee$   |  $\wedge$   |      $\uplus$      |
|   \cup   |  \cap   |  \sqcup   |  \sqcap   |    \vee    |   \wedge    |       \uplus       |

## 6、大运算符

|   $\sum$   | $\bigoplus$  |  $\prod$  | $\int$  | $\bigcup$ |  $\bigvee$  | $\bigsqcup$ |
| :--------: | :----------: | :-------: | :-----: | :-------: | :---------: | :---------: |
|    \sum    |  \bigoplus   |   \prod   |  \int   |  \bigcup  |   \bigvee   |  \bigsqcup  |
| $\bigodot$ | $\bigotimes$ | $\coprod$ | $\oint$ | $\bigcap$ | $\bigwedge$ | $\biguplus$ |
|  \bigodot  |  \bigotimes  |  \coprod  |  \oint  |  \bigcap  |  \bigwedge  |  \biguplus  |

## 7、三角运算符

| $\bot$ | $\angle$ | $30^\circ$ | $\sin$ | $\cos$ | $\tan$ | $\cot$ | $\sec$ | $\csc$ |
| :----: | :------: | :--------: | :----: | :----: | :----: | :----: | :----: | :----: |
|  \bot  |  \angle  |  30^\circ  |  \sin  |  \cos  |  \tan  |  \cot  |  \sec  |  \csc  |

## 8、微积分运算符

| $\prime$ | $\int$ | $\iint$ | $\iiint$ | $\oint$ | $\lim$ | $\infty$ | $\nabla$ |
| :------: | :----: | :-----: | :------: | :-----: | :----: | :------: | :------: |
|  \prime  |  \int  |  \iint  |  \iiint  |  \oint  |  \lim  |  \infty  |  \nabla  |

## 9、逻辑运算符

| $\because$ | $\therefore$ | $\forall$ | $\exists$ | $\not=$ | $\not>$ | $\not\subset$ |
| :--------: | :----------: | :-------: | :-------: | :-----: | :-----: | :-----------: |
|  \because  |  \therefore  |  \forall  |  \exists  |  \not=  |  \not>  |  \not\subset  |

## 10、戴帽符号

|  $\hat{y}$  | $\check{y}$ |   $\tilde{y}$   |
| :---------: | :---------: | :-------------: |
|   \hat{y}   |  \check{y}  |    \tilde{y}    |
| $\grave{y}$ |  $\dot{y}$  |   $\ddot{y}$    |
|  \grave{y}  |   \dot{y}   |    \ddot{y}     |
|  $\bar{y}$  |  $\vec{y}$  |  $\widehat{Y}$  |
|   \bar{y}   |   \vec{y}   |   \widehat{Y}   |
| $\acute{y}$ | $\breve{y}$ | $\widetilde{Y}$ |
|  \acute{y}  |  \breve{y}  |  \widetilde{Y}  |

## 11、连线符号

| $\overline{a+b}$ | $\underline{a+b}$ | $\overbrace{a+\underbrace{b+c}_{1.0}+d}^{2.0}$ |
| :--------------: | :---------------: | :--------------------------------------------: |
|  \overline{a+b}  |  \underline{a+b}  |  \overbrace{a+\underbrace{b+c}_{1.0}+d}^{2.0}  |

## 12、箭头符号

|  $\uparrow$  |  $\Uparrow$  | $\rightarrow$ | $\Rightarrow$ | $\longrightarrow$ | $\Longrightarrow$ |
| :----------: | :----------: | :-----------: | :-----------: | :---------------: | :---------------: |
|   \uparrow   |   \Uparrow   |  \rightarrow  |  \Rightarrow  |  \longrightarrow  |  \Longrightarrow  |
| $\downarrow$ | $\Downarrow$ | $\leftarrow$  | $\Leftarrow$  | $\longleftarrow$  | $\Longleftarrow$  |
|  \downarrow  |  \Downarrow  |  \leftarrow   |  \Leftarrow   |  \longleftarrow   |  \Longleftarrow   |