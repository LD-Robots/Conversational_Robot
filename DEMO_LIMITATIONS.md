# 🚧 Limitări Tehnice & Stadiu Proiect (Versiune Demo)

> **Context:** Acesta este un prototip funcțional (Proof of Concept) care rulează hybrid: procesare vocală locală (pe laptop CPU) și inteligență artificială în cloud (Groq). Arhitectura este optimizată pentru flexibilitate, nu încă pentru viteză comercială instantanee.

---

## 1. Viteză și Latență (De ce durează uneori?)
Timpul de răspuns variază semnificativ în funcție de complexitatea întrebării:
*   **Întrebări Simple:** ("Ce capitală are Franța?") → Răspuns rapid (~2-4 secunde).
*   **Întrebări cu Web Search:** ("Cum e vremea azi?" sau "Cine a câștigat meciul aseară?")
    *   *Limitare:* Robotul trebuie să "gândească", să caute pe Google, să citească rezultatele și să sintetizeze răspunsul.
    *   *Impact:* Latența poate crește la **8-15 secunde**. Este un compromis necesar pentru a avea date actualizate.
*   **Procesare Locală:** Deoarece conversia Voce-Text (ASR) și Text-Voce (TTS) se fac pe procesorul laptopului (CPU), viteza depinde și de ce alte programe sunt deschise.

## 2. Constrângeri Hardware (CPU vs GPU)
*   **Fără GPU Dedicat:** Proiectul este construit să ruleze pe hardware accesibil (laptop standard). Sistemele comerciale (Alexa/Siri) folosesc servere gigantice pentru viteză instantanee. Noi demonstrăm că se poate și local, dar cu un cost minor de viteză.
*   **Consum Resurse:** În timpul procesării vocii, ventilatoarele laptopului pot porni. E normal.

## 3. Limitări Audio & Vocale
*   **Accente Mixte:** Motorul de voce (TTS) se setează pe o singură limbă per frază (Română SAU Engleză).
    *   *Efect:* Dacă robotul spune un nume englezesc într-o frază românească (ex: "Mergem la *New York*"), va pronunța "New York" cu accent românesc. Este o limitare a tehnologiei de sinteză actuale.
*   **Barge-in (Întreruperea):** Puteți întrerupe robotul în timp ce vorbește, dar sistemul de anulare a ecoului este software.
    *   *Sfat:* Vorbiți ferm și clar pentru a-l "tăia". Dacă muzica e tare sau e gălăgie, s-ar putea să nu vă audă întreruperea.

## 4. Dependența de Internet
*   Deși "urechile" și "gura" robotului sunt locale (offline), "creierul" (Groq AI) necesită conexiune la internet.
*   Dacă internetul este instabil, robotul poate părea că "ascultă" dar nu răspunde (timeout la request-ul către server).

## 5. Stabilitate (Software în Dezvoltare)
*   Fiind un proiect de cercetare/dezvoltare, pot apărea:
    *   Ocazional, robotul poate "auzi" liniște și să creadă că ați terminat de vorbit prea repede.
    *   Halucinații AI (răspunsuri foarte convingătoare, dar factually greșite), specific tuturor modelelor de limbaj (LLM) actuale.

---

### 🎯 Concluzie
Demonstrăm un asistent vocal **bilingv**, capabil de **căutare pe internet** și **dialog natural**, integrat complet pe un laptop obișnuit. Limitarea principală este latența variabilă indusă de complexitatea task-urilor (Web Search) și lipsa accelerării hardware dedicate.
