---
title: SimCLR
tags: []
id: '1521'
categories:
  - - ML/DL學習
date: 2021-11-27 22:22:09
---

<img src="https://i.imgur.com/1mfXZ7M.jpg" class="center">

**_本文純為學術分享，內文圖片皆來自原始論文(封面圖片：Photo by [Pietro Jeng](https://unsplash.com/@pietrozj?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/deep-learning?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) )_**

**_Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020, November). A simple framework for contrastive learning of visual representations. In International conference on machine learning (pp. 1597-1607). PMLR.。[https://arxiv.org/abs/2002.05709](https://arxiv.org/abs/2002.05709)_**

要如何真正學習到image representation並達到task agnostic的模型，是當前一個頗熱門的研究主題，這篇論文雖然有點久遠，是2020年中的一篇文章，但是當初提到的基本概念，到現在已經發展成了像當紅的CLIP(Contrastive Language-Image Pretraining)，因此我想說也可以來review一下這篇目前google scholar引用次數已經來到2600多次的文章：A Simple Framework for Contrastive Learning of Visual Representation. 不過這篇的主要目的依然是筆記，因此關於更多詳細的內容，還是請讀者回去翻翻原始論文喔(傳送門：[https://arxiv.org/abs/2002.05709](https://arxiv.org/abs/2002.05709 "https://arxiv.org/abs/2002.05709"))~

<!-- more -->

## Method

簡單來說，作者實現學習visual representation的作法，是利用unsupervised learning的方式，加上contrastive loss的概念，讓模型自動學會哪些representation是屬於同一類的，並將這樣的學習結果帶到不同類型的task上面，得到一個好的pre-train model。至於細節的部分，還是先讓我們看看論文裡面的framework架構：

![](https://i.imgur.com/wfWOUpP.png)

首先一個x代表一張圖片，每張圖片都會通過兩個不同的data augmentation的function(這裡用t和t'表示)，得到xi和xj，為了接下來解說方便，我們先考慮左邊xi的那條branch，右邊xj基本上是同樣的概念。於是xi又會經過ResNet(這裡可以直接把這層ResNet當作一層encoder)，在通過average pooling之後，進入到FC之前，我們直接把feature vector(hi)取出來過一個network projection head(g(.))，把hi project到contrastive learning的latent space。這個g(.)包含了兩層的weight中間夾一個ReLU activation，相當於一個2-layer MLP，最後得到zi。同理經過右邊的branch我們也可以得到另一種data augmentation生成的zj。緊接著就是重頭戲了，我們要利用zi和zj去計算他們的contrastive loss。作者這邊用的loss function概念應該就是常見contrastive loss的function，作者把它叫做NT-Xent，我們可以直接看算式說故事：

![](https://i.imgur.com/GbV1784.png)

NT-Xent的目的，是要讓同一張圖片生成的zi和zj愈相似，其他的不相干的圖片拉得愈遠愈好。所以可以看到分子是同一張圖片zi和zj的cosine similarity，分母則是zi和任一張不同來源圖片的zk的cosine similarity，由於log前面是負號，minimize loss function的結果會讓分子愈來愈大，分母愈來愈小，達到"同一張圖片生成的zi和zj愈接近，其他的不相干的圖片拉得愈遠愈好"的訓練效果。其中一個小細節是分母的大寫1代表一個indicator vector，在k不等於i的時候才會等於1，所以zi本人不會跟自己算cosine similarity啦(廢話)。而r代表temperature parameter(就只是normalization的一個參數)，論文裡看起來應該是fixed的而不是learnable的。

更細節的部份我們可以觀賞論文裡的pseudo code，看完這裡應該就可以完整了解整個模型的架構了：

![](https://i.imgur.com/ACev4yW.png)

總之我們可以先把training data切成未知個N size的 batch，每個batch裡1~N的圖片都給他做兩種不同的data augmentation，得到z2k-1和z2k，而這裡的si,j就是cosine silmilarity。接著實際上的loss function定義為大L那條式子，除了考慮l(2k-1, 2k)，也要考慮l(2k, 2k-1)，因為單用小l的loss在分母那邊會少考慮到反向的結果。最後return的是f(.)以前的encoder架構，g(.)只是用來pre-train訓練，在實際應用的時候不會用到~(作者針對這樣的做法有做詳細的實驗，待會會提)

好啦到這裡應該就已經大致講完整個架構了，接下來就是一些對於architecture的實驗和testing的結果。

## Discussion & Results

首先我們剛剛都沒有講說作者使用了哪些data augmentation，作者實驗了Crop, filp, Cutout, Color distort, Rotate, Gaussian noise, Gaussian blue等方式， 並將實驗後得到的feature network(也就是f(.)以前的架構)，在ImageNet dataset上做linear probe，實驗結果會發現，單用一種類的data augmentation效果並不是很好。可見下圖：

![](https://i.imgur.com/gjuirvp.png)

作者舉了一個例子，如果data augmentation同樣都用crop可能就會出現嚴重的問題，如果畫出一張圖片pixel intensities histogram，就會發現即使random crop一張圖片的不同區域，得到的pixel intensities分布也會幾乎一樣，於是data augmentation是沒有效果的，所以才需要兩種modalities以上的augmentation。而最後作者挑選了三種augmentation方式：random crop(with flip and resize), color distortion和Gaussian blur。

![](https://i.imgur.com/JoOIiod.png)

第二，作者實驗了256~8192的batch size，發現 large batch size因為增加了單一batch當中negative pairs的數量，所以會提升model在top-1 accuracy的表現，這樣的現象在supervised learning就不會特別明顯。

第三，也是我認為滿重要的地方，是projection head的作用，以及為甚麼要丟掉g(.)學到的訊息。作者提到，如果比較projection head之前得到的h=f(t)，和加上projection head後得到的z=g(h)，兩者拿去做linear probe，前者的accuracy比後者高很多。作者提出的假說是，projection head會造成information loss，尤其是那些data transformation的資訊，所以不能拿z=g(h)去做linear probe。於是，作者就把h和g(h)直接拿出來做training來predict各張圖片做過甚麼transformation以驗證自己的假說，實驗結果真的發現g(h)學習出來的效果較差。不過我感覺這是一個值得研究的問題，比如說搞不好projection head丟失的不只有data transformation的資訊，有沒有別的方法可以設計這個projection head，感覺都是可以再考慮的地方。

最後還有一些testing的結果沒有看到，總之作者在ILSVRC-12上面做了semi-supervised learning(sample 1%和10%的labeled dataset做training)的實驗，結果SimCLR超越了SOTA的accuracy，另外如果用整個ImageNet的dataset fine-tuned在ResNet上面，不僅沒有catastrophic forgetting的問題(作者應該是沒有做regularization)，更outperform了SOTA的效果(SOTA可是直接train from scratch)，這個結果也是滿有趣的。

同時作者也在其他12個dataset上做transfer learning，採用了linear probe和fine-tuning兩種方法，SimCLR的表現如下表，其實表現得都滿不錯的

![](https://i.imgur.com/jYoeZGk.png)

感謝大家看到這裡，我其實省略了很多細節和實驗結果，主要還是希望可以把重點抓出來，也讓自己的思緒更清晰。希望有興趣的人會喜歡~~

## 參考資料

1.  Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020, November). A simple framework for contrastive learning of visual representations. In _International conference on machine learning_ (pp. 1597-1607). PMLR.