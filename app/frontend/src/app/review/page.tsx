import { BatchReviewWorkspace } from "@/features/calibration/BatchReviewWorkspace";

export const metadata = { title: "Полевая проверка 65 кадров" };

export default function ReviewPage() {
  return (
    <main className="inner-page inner-page--wide review-page">
      <header className="page-intro page-intro--row">
        <div><span className="eyebrow">полевой контроль</span><h1>Проверка<br />65 кадров</h1></div>
        <p>Полная очищенная полевая выборка. Два неоднозначных/OOD-кадра вынесены отдельно и не участвуют в known-метриках.</p>
      </header>
      <BatchReviewWorkspace />
    </main>
  );
}
