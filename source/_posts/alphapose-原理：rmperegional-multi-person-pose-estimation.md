---
title: Alphapose 原理：RMPE(Regional Multi-Person Pose Estimation)
tags: []
id: '1537'
categories:
  - - ML/DL學習
date: 2021-12-05 16:41:55
---

<img src="https://i.imgur.com/Lq57wix.jpg?1" class="center">

**_本文純為學術分享，內文圖片皆來自原始RMPE論文與另一篇STN的論文(封面圖片：Photo by [Patricia Palma](https://unsplash.com/@laclem?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/pose?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText))_**

**_Fang, H. S., Xie, S., Tai, Y. W., & Lu, C. (2017). Rmpe: Regional multi-person pose estimation. In Proceedings of the IEEE international conference on computer vision (pp. 2334-2343)._** **_[https://arxiv.org/abs/1612.00137](https://arxiv.org/abs/1612.00137)_**

**_Jaderberg, M., Simonyan, K., & Zisserman, A. (2015). Spatial transformer networks. Advances in neural information processing systems, 28, 2017-2025._** _**[https://arxiv.org/abs/1506.02025](https://arxiv.org/abs/1506.02025 "https://arxiv.org/abs/1506.02025")**_

現在的pose estimation已經有很多模型或是技術可以實作，而在這之中，又不能不提到Alphapose。即便發表於久遠的2016年，Alphapose 完整的open source API和不錯的accuracy依舊是現在許多CV研究者的好幫手，而這樣好用的API，背後原理的全名其實是RMPE(Regional Multi-Person Pose Estimation)，不過這個名字大家可能也加減看看就好，因為翻遍整篇論文，這個名詞幾乎指出現在introduction和conclusion的幾行而已 XD。總之，今天要來review的就是這篇RMPE: Regional Multi-Person Pose Estimation。

<!-- more -->

## Introduction

一般來說pose estimation分成兩種：part-based framework和 two-step framework，前者會先辨識身體的關節點，再重組這些關節點的關係預測human pose；後者會先detect human bounding box，再根據bounding box的位置去辨識human pose，RMPE也是採用這類方法。然而這類方法有一個缺點，那就是如果bounding box的位置不準，那麼pose estimation的效果也會大打折扣，為了解決這個問題，作者提出的RMPE有三個革新的架構：(1)Symmetric STN and parallel SPPE (2)Parametric pose NMS (3)Pose-guided proposals generator，接下來就會針對這三個架構詳加討論

## Model Architecture

![](https://imgur.com/R6KblWC.png)

圖一

整個RMPE的大架構如上圖：首先input image會先通過現有的Human detector得到human proposals，接著這些human proposals會分別通過剛剛說到的STN(spatial transformer network)、SPPE(single person pose estimation)、SDTN(spatial de-transformer network)，經過這三個步驟之後，我們就會從human detector框出的一個個人身上預測到他們的pose，然而human proposal有可能會重複而非一一對應，於是我們要把同一個人多出來的pose estimation刪掉，透過的方法便是pose NMS，最後便可以完成對一張圖片的pose estimation。

### Symmetric STN and parallel SPPE

首先是SSTN和parallel SPPE的架構，這裡SSTN的全名是symmetric spatial transformer network，但這裡指的transformer並不是我們常聽到跟self- attention有關的那個transformer，而是另外一個如下圖的架構：

![](https://i.imgur.com/O76sbGh.png)

圖二

STN的目的是要透過空間映射的方式，讓不一定在畫面中間的人回到畫面的中間，這裡的input應該是一張含有human proposal的圖片，然後透過Localisation network，可以學到affine transformation的矩陣θ，接著利用這個θ，我們可以把原本在human proposal上的座標進行affine transformation，這個過程就是grid generator，接著再進入sampler，sampler主要的目的則是要解決整數座標不能為分的問題(Ex: 如果Τθ(G)輸出的是一個非整數的值，那要怎麼對應到座標，是要四捨五入嗎，可是四捨五入就不能微分了？)

這裡不會仔細解釋sampler的數學證明，不過我們還是可以觀賞一下STN原文中的公式：

![](https://i.imgur.com/bkHA4R2.png)

圖三

![](https://i.imgur.com/k0NjwT1.png)

圖四

上面的式子是generalized form，下面的式子則是假設了一個k的函數。xi和yi可以想成是grid generator的output座標，我們要採用線性差值的技巧，抓出在xi、yi附近的點(n,m)(採用max函數，所以太遠的點就不會用到)，用點(n,m)的score(U)，對(xi、yi)的score做估計，而這樣的函數就會是一個可以微分的式子。

好吧繞得有點遠，我們應該要回到STN的θ，還記得我們要用θ把我們圖片上的點做空間映射，論文裡面花了滿大篇幅在解釋這件事情：

![](https://i.imgur.com/AfzeB44.png)

圖五

\[θ1 θ2 θ3\]是一個2x3的矩陣，θ1、θ2主要是負責做旋轉、放大縮小等轉換，而θ3則是負責平移。

而既然我們將座標映射到了另一個空間，那麼最後輸出預測的時候就要再把它映射回來，於是整個SSTN的架構其實是這樣的：

![](https://i.imgur.com/ObGdWrw.png)

圖六

除了正向的STN，還要用一個SDTN(Spatial de-transformer network)把座標映射回原本的圖片)，於是de-transformer的公式如下：

![](https://i.imgur.com/AE4CgEa.png)

而\[γ1、γ2\]就是\[θ1、θ2\]的反矩陣，因為反矩陣一定是一個方陣，所以γ3要另外處理：

![](https://i.imgur.com/fceGz47.png)

這裡我做了一個簡單的推導：

![未提供說明。](https://scontent.ftpe7-3.fna.fbcdn.net/v/t1.15752-9/262915587_314501503677681_7771690257437202969_n.jpg?_nc_cat=102&ccb=1-5&_nc_sid=ae9488&_nc_ohc=Z8Lid5vdnVwAX9aHYCm&_nc_ht=scontent.ftpe7-3.fna&oh=7f5079e3a55a12652ab9b5b8d3cbfa12&oe=61D11731)

圖七

於是有了這些條件，我們就可以盡情地利用back-propagation更新θ參數了：

![](https://i.imgur.com/gtbGP2G.png)

![](https://i.imgur.com/7Oox7nG.png)

再來要討論的是parallel SPPE，理論上STN的output會通過SPPE去做pose estimation，但是這樣還不夠，我們除了原本的SPPE，又加上了另一個SPPE，也就是作者propose的parallel SPPE，多出來的這個SPPE(我們簡稱SPPE-2)並不會進入SDTN，而SPPE-2 output出來的pose，會直接和center-located ground truth做比較，也就是這邊的ground truth全部都是center-located的。還記得STN的目的嗎？就是為了要讓human pose center-located，所以利用SPPE-2，我們可以在training的時候固定SPPE-2的參數，讓loss function back propagate回去更新STN的參數，也同時更新SDTN的參數(還記得SDTN的參數可以直接用θ推導)，讓STN更容易學習到如何讓human pose跑到圖片中間，優化真正SPPE的預測效果。

![](https://i.imgur.com/PSziJ6s.png)

圖八

大家可能會疑問為甚麼要設計這樣parallel的網路，而不是直接在原本的SPPE後面加上一個與center-located ground truth的loss就好了，論文也為大家解了惑，原因是STN transformation的效果有限，沒辦法把人真的放到畫面中間，於是如果在SPPE後面加上這樣的loss並同時更新SPPE的參數，會影響到原本SPPE的效果。也由於STN不完美的特性，parallel SPPE能一直傳遞大量的loss優化STN的架構。

### Parametric Pose NMS

前面提過human detector常常會偵測到redundant的human proposal，多出來的這些proposal餵進去SPPE之後就會產生一樣多餘的pose estimations，為了解決這個問題，我們必須要能夠判斷該刪掉哪些多餘的pose。作者設計了一個NMS的criterion (non-maximum suppression)，概念上就是，我們判斷這兩個pose的位置是不是夠靠近，如果夠靠近的話，那我們就得把其中一個pose刪掉。

於是我們可以先來看看作者定義的pose distance的公式：

![](https://i.imgur.com/nw0B3ck.png)

![](https://i.imgur.com/hAHYbXn.png)

Ksim定義

![](https://i.imgur.com/odUBrLB.png)

Hsim定義

Pi、Pj代表兩個不同的pose estimation，ci、cj則分別是這兩個pose的confidence score，Ksim和Hsim的定義則分別如上：

首先看看Ksim，我們假設Bi是Pi的bounding box，B(kin)則是以第i個pose的第n個關節點為中心所畫出的bounding box，其寬高分別為Bi的1/10倍。Ksim所代表的意思，就是我們以Pi為reference，把Pi當中每個關節點都取出來和Pj相對應的每個關節點比較。第一件事是看看kjn這個關節點有沒有落在kin的bounding box裡面，有的話就往下計算，沒有的話這兩個點的similarity score就直接設為0，接下來如果kjn在kin的bounding box裡面，則分別把他們兩個的confidence score送進tanh函數之後再相乘，最後把所有pair的分數加起來，得到最後的Ksim值。

接下來是Hsim，Hsim就比較好理解了，就是計算kin和kjn之間的距離，只不過這個距離是用一個RBF function去做計算。

最後把Ksim與Hsim加起來，再套上一個校正的權重λ，就得到作者定義的pose distance公式。

利用這個pose distance公式，就可以推得作者設計的NMS criterion：

![](https://i.imgur.com/f88p0e3.png)

其中∧、λ代表的只是計算pose distance時會用到的parameter，η則是一個被定義的threshold，如果pose distance小於η，則這個indicator vector就會給它1的值，1代表的是這個pose必須被eliminated。

另外作者有特別強調elimination criterion當中的四個參數：σ1、σ2、η、λ都是可以optimize的，而optimization的方式沒有特別敘述，有可能是傳統的grid search或是random search之類的。

### Pose-guided Proposals Genreator

事實上two-stage pose estimation很容易受到稍微偏差的human proposal影響，所以作者設計了一個data augmentation的方式，也為他取了一個很炫砲的名字Pose-guided Proposals Genreator(PGPG)，這個PGPG做的事情，是要找出各個pose當中，detected bounding box與ground truth bounding box的offset distribution。簡單來說，bounding box都有四個頂點：(xmin, xmax, ymin, ymax)，我們可以比對每個detected bounding box與其對應的ground truth bounding box，在這四個點上的距離差，並用高斯模型估計，得到四個分布圖，藉由這些分布，我們就可以做隨機取樣，在原圖上取樣出好幾倍的human proposals，並把他們送進去模型一起訓練。

![](https://i.imgur.com/FYn1srn.png)

圖九

## Expreiments and Results

終於進入到results了，作者主要是在兩個Dataset上進行evaluation，分別是MPII和MSCOCO，在testing的時候作者選擇VGG作為human detector，每個human proposal還會在把它們的長寬加大30%以免沒框到人。至於SPPE的部分則是採用stack hourglass model，這是不會細講stack hourglass model，但它主要是一個多層residual CNN的架構，由於feature map size會先變小，再upsampling恢復原狀，因此看起來很像某種沙漏的形狀。事實上除了選擇上述模型，作者也嘗試了其他模型，以證明RMPE的方法是可以被泛化的。

![](https://i.imgur.com/XuwhV8Z.png)

圖10

![](https://i.imgur.com/7bNJaOB.png)

圖11

上面兩張圖分別代表MPII和MSCOCO測試的結果，與當時的模型比較，都是state-of-the-art的performance。

### Ablation study

Ablation study的部分，作者分別拿掉了Symmetric STN & Parallel SPPE、Parametric Pose NMS、Pose-guided Proposals Generator這三個主要架構，也都發現對實驗結果有明顯的影響，其中比較有趣的是，在Parametric Pose NMS的實驗裡，作者比較了random jittering(字面上的意思看起來像是隨便取樣)和PGPG的效果，結果發現PGPG還是比較好，因此data augmentation還是要有特定的方式才能提升performance。另外作者他們也認為他們parametric Pose NMS會比別人好，是因為他們的參數可以optimize，過去的方法都沒有對NMS的參數做optimization。

總之這是一篇很有趣的論文，作者用了許多fancy的方法來建構他的模型，也因此可能有比較多的小細節沒有辦法全部含括在文章裡，有興趣的人可以翻翻原文或是相關資料，最後還是感謝大家看到這裡，希望有興趣的人會喜歡~~

## Reference

1.  **_Fang, H. S., Xie, S., Tai, Y. W., & Lu, C. (2017). Rmpe: Regional multi-person pose estimation. In Proceedings of the IEEE international conference on computer vision (pp. 2334-2343)._** **_[https://arxiv.org/abs/1612.00137](https://arxiv.org/abs/1612.00137)_**
2.  **_Jaderberg, M., Simonyan, K., & Zisserman, A. (2015). Spatial transformer networks. Advances in neural information processing systems, 28, 2017-2025._** _**[https://arxiv.org/abs/1506.02025](https://arxiv.org/abs/1506.02025)**_