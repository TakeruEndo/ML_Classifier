# ペンギンの分類(18クラス分類)
## dataset
bingから収集  
train/test: 2218/89  

## CNNモデル
1. SEResNet
```
epoch: 0  {'train_loss': 0.7354396602937153} {'valid_loss': 2.068526824315389, 'val_acc': tensor(0.5281, device='cuda:0')}
epoch: 1  {'train_loss': 0.5696406903011458} {'valid_loss': 2.4503137270609536, 'val_acc': tensor(0.5056, device='cuda:0')}
epoch: 2  {'train_loss': 0.4333643302321434} {'valid_loss': 2.071386913458506, 'val_acc': tensor(0.5506, device='cuda:0')}
epoch: 3  {'train_loss': 0.3016752179179873} {'valid_loss': 1.6000480651855469, 'val_acc': tensor(0.6180, device='cuda:0')}
epoch: 4  {'train_loss': 0.33400224191801886} {'valid_loss': 2.222716808319092, 'val_acc': tensor(0.5506, device='cuda:0')}
epoch: 5  {'train_loss': 0.1846762572016035} {'valid_loss': 1.8656799793243408, 'val_acc': tensor(0.5955, device='cuda:0')}
epoch: 6  {'train_loss': 0.11048330409186227} {'valid_loss': 1.8652525742848713, 'val_acc': tensor(0.6966, device='cuda:0')}
```

## 結果の例
![result](https://github.com/TakeruEndo/ML_Classifier/blob/master/penguin_search/images/demo.png)

## ペンギンの種類
1. Emperor Penguin
2. Rockhopper Penguin
3. Galápagos penguin
4. King Penguin
5. White-flippered Penguin
6. Humboldt Penguin
7. Magellanic Penguin
8.  African Penguin
9.  Little Penguin
10. Chinstrap Penguin
11. Fiordland penguin
12. Erect-crested Penguin
13. Snares Islands penguin
14. Royal Penguin
15. Macaroni Penguin
16. Adelie Penguin
17. Gentoo penguin
18. Yellow-eyed penguin

### 参考
https://penguin-book.com/category/species/

## Bing Image Search API v7
### 参考
https://qiita.com/m-shimao/items/74ee036fff8fac01566e