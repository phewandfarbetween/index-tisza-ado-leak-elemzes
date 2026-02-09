# Az állítólagos adótervezet és a hivatalos program szerzőségének elemzése

## TL;DR

Az Indexen közzétett dokumentumot objektíven mérhető kritériumok alapján minden bizonnyal mesterséges intelligenciával generálták.

**Frissítés**: a [Tisza Párt valódi programja](https://magyartisza.hu/program) ugyanezen kritériumok alapján emberi alkotás.

## Részletek

Az Index 2025. december 1-én közzétett egy többszáz oldalas adótervezetet, ami állítólag a Tisza Párthoz kötődik.

Számos jel utal arra, hogy a dokumentumot [mesterséges intelligenciával generálták](https://www.reddit.com/r/hungary/comments/1pb9qen/bizony%C3%ADt%C3%A9kok_hogy_a_tisza_ad%C3%B3terveit_tartalmaz%C3%B3/), többek között:

 * jellegzetes stílus és megfogalmazások,
 * formázási és tördelési bakik,
 * utalások az MI számára adott feladatkiírásra (ún. prompt), pl. "_Hatás az államháztartásra — részletes indoklással_",
 * stb.

Azonban a teljes bizonyosság kedvéért hasznosnak tartom objektív, számszerűsíthető bizonyítékok alapján is megvizsgálni a dokumentumot.

**Frissítés**: 2026. február 7-én a Tisza Párt közzétette a [programját](https://magyartisza.hu/program), ezért kíváncsiságból erre is elvégeztem ugyanezeket a vizsgálatokat.

### LLM-mel generált szövegek lebuktatása

A [nagy nyelvi modellek (LLM-ek)](https://hu.wikipedia.org/wiki/Nagy_nyelvi_modell) által generált szövegek matematikailag tekinthetők egy olyan valószínűségi eloszlásból vett véletlenszerű mintának, ami az adott LLM-re illetve LLM-családra jellemző.  Minden bizonnyal ez áll a hátterében az LLM-ek azon képességének, hogy [fel tudják ismerni a sajátmaguk illetve a hozzájuk hasonló LLM-ek által generált szövegeket](https://arxiv.org/abs/2410.21819).

Az ilyen feladatokban egy-egy konkrét szöveg esetén látszólag csak véletlenszerűen találgatnak, de nagyobb számú kísérlet összegzéséből kiderül, hogy a véletlenszerű találgatásnál általában jelentősen jobb eredményt érnek el.

### Az állítólagos adótervezet és a tényleges program vizsgálata

 1. Az Indexen közzétett PDF dokumentumok - feltételezhetően az elemzés megnehezítése céljából - a szöveget képként tartalmazzák, ezért először [OCR](https://hu.wikipedia.org/wiki/Optikai_karakterfelismer%C3%A9s) segítségével szöveggé alakítottam őket. Az így kapott, karakterfelismerési hibáktól hemzsegő szövegből véletlenszerűen kimásoltam 25 darab hosszabb-rövidebb részletet. (`leak-samples-ocr.txt`)

 2. Az összehasonlítás kedvéért összegyűjtöttem 25 darab olyan pénzügyi témájú szövegrészletet, amiket minden bizonnyal emberek írtak MI használata nélkül, de hasonlóan tárgyilagos fogalmazási stílusban. Vigyáztam arra, hogy ezek viszonylag újak legyenek, azaz ne szerepeljenek még a népszerű nyelvi modellek tanításához használt korpuszokban. (MNB és KSH tanulmányok részletei, G7 cikkrészletek, stb.) (`human-samples.txt`)

 3. LLM-ek segítségével generáltam 25 darab esszét illetve törvényjavaslatot adóemelésekről, nyugdíjcsökkentésekről, energiaár-emelésekről, stb. és ezekből is elmentettem részleteket. (`llm-samples.txt`)

 4. A pontosabb összehasonlítás érdekében a fenti tiszta szövegekből készítettem véletlenszerű OCR-hibákat tartalmazó változatokat is a tényleges OCR-hibák alapján. (`human-samples-ocr.txt` és `llm-samples-ocr.txt`)

 5. Az így kapott mintákat egyenként [értékeltettem](https://en.wikipedia.org/wiki/LLM-as-a-Judge) 4 különböző LLM-mel ötfokozatú skálán aszerint, hogy az adott szövegrészlet milyen mértékben tartalmazhat ember illetve LLM által írt szöveget. (1 pont = 100% emberi szöveg, 5 pont = 100% gép által írt szöveg.)

     * A prompt minden esetben ugyanazokat az instrukciókat tartalmazta, tehát az egyetlen változó a vizsgálandó szövegrészlet volt.

     * 25 minta már statisztikailag értelmezhető, de még nem kerül néhány dollárnál többe az elemzése a népszerű LLM-szolgáltatóknál.

     * Az értékeléshez az ún. "temperature" paramétert alacsony, 0.1-es értékre állítottam be, hogy növeljem az LLM-bírák determinisztikusságát.

 6. Az egyes minták átlagpontszámait [Mann-Whitney-próbával](https://hu.wikipedia.org/wiki/Mann%E2%80%93Whitney-pr%C3%B3ba) hasonlítottam össze, hogy megtudjam, az állítólagos adótervezet mintái az LLM-ek vagy az emberek által írt szövegekhez hasonlítanak-e jobban.

 7. Az ismert eredetű minták pontszámaira logisztikus regresszió modellt is illesztettem, majd ezzel kategorizáltam a vizsgálandó mintákat.

 8. A Tisza Párt által közzétett programból is gyűjtöttem 25 darab mintát (`tisza-samples.txt`), valamint ezekből is készítettem egy szimulált OCR-hibákat tartalmazó változatot (`tisza-samples-ocr.txt`), és ezekre is elvégeztem a fenti teszteket.

 9. [u/Kukaac javasolt egy promptot](https://www.reddit.com/r/hungary/comments/1qz6o4b/statisztikai_m%C3%B3dszerekkel_elemeztem_a_tisza_p%C3%A1rt/o4enr5i/), amivel szerinte a ChatGPT tud olyan szöveget generálni, ami átveri a tesztet. A kedvéért generáltattam a ChatGPT-vel ezzel a prompttal is 25 mintát (`kukaac-samples.txt`).

    > Generálj egy 20 soros rövid essayt az AI hatásáról az oktatásra. Használj vegyesen rövid és hosszú, összetett mondatokat. Használj szubjektív véleményeket. Kerüld a túl steril, tankönyvi megfogalmazást.

### Eredmény

Éles különbség rajzolódik ki az emberek által írt és az LLM-mel generált szövegek pontszámai között, és az állítólagos adótervezet erősen az utóbbi csoporthoz áll közel, míg a valódi program az előbbiekhez:

```
Pontszámok átlaga (1 = biztosan ember, 5 = biztosan LLM)

  human-samples.txt:        1.960  (std: 1.509)
  human-samples-ocr.txt:    2.180  (std: 1.584)
  llm-samples.txt:          4.110  (std: 1.363)
  llm-samples-ocr.txt:      4.200  (std: 1.288)
  leak-samples-ocr.txt:     3.910  (std: 1.408)
  tisza-samples.txt:        1.900  (std: 1.261)
  tisza-samples-ocr.txt:    2.300  (std: 1.507)
  kukaac-samples.txt:       3.770  (std: 1.624)

## leakocr

  Mann-Whitney-próba
    leakocr vs humanocr: p-value = 0.000000066
    leakocr vs llmocr:   p-value = 0.109163194
    A leakocr minta megkülönböztethetetlen az llmocr mintától és különbözik a humanocr mintától.

  LogisticRegression
    leakocr átlagos LLM valószínűség: 74.479%  (std: 0.292)
    leakocr medián LLM valószínűség:  92.302%

## tisza

  Mann-Whitney-próba
    tisza vs human: p-value = 1.000000000
    tisza vs llm:   p-value = 0.000000001
    A tisza minta megkülönböztethetetlen a human mintától és különbözik az llm mintától.

  LogisticRegression
    tisza átlagos LLM valószínűség: 4.115%  (std: 0.069)
    tisza medián LLM valószínűség:  1.351%

## tiszaocr

  Mann-Whitney-próba
    tiszaocr vs humanocr: p-value = 0.433814392
    tiszaocr vs llmocr:   p-value = 0.000000002
    A tiszaocr minta megkülönböztethetetlen a humanocr mintától és különbözik az llmocr mintától.

  LogisticRegression
    tiszaocr átlagos LLM valószínűség: 10.593%  (std: 0.175)
    tiszaocr medián LLM valószínűség:  1.344%

## kukaac

  Mann-Whitney-próba
    kukaac vs human: p-value = 0.000000030
    kukaac vs llm:   p-value = 0.157949508
    A kukaac minta megkülönböztethetetlen az llm mintától és különbözik a human mintától.

  LogisticRegression
    kukaac átlagos LLM valószínűség: 74.223%  (std: 0.295)
    kukaac medián LLM valószínűség:  96.017%
```

<img src="https://raw.githubusercontent.com/phewandfarbetween/index-tisza-ado-leak-elemzes/main/boxplot.png" alt="Boxplot"/>

Ezek alapján kijelenthető, hogy az állítólagos adótervezet szövegét LLM-ekkel generálták, a Tisza Párt valódi programja pedig emberi munkával készült, u/Kukaac promptja pedig nem tudja kijátszani a tesztet.
