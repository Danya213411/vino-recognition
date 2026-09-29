import { ReviewMatrixWorkspace } from "@/features/calibration/ReviewMatrixWorkspace";

export const metadata = {
  title: "Матрица ошибок распознавания",
  description: "Исходные кадры, Top-5 кандидатов и полный разбор OCR для полевых ошибок.",
};

export default function MatrixPage() {
  return (
    <main className="matrix-page">
      <header className="matrix-intro">
        <div>
          <span className="eyebrow">разбор последнего прогона</span>
          <h1>Матрица ошибок</h1>
        </div>
        <p>Только кадры, где правильное вино оказалось не на первом месте. Исходник сопоставлен со всем Top‑5 и сырыми данными OCR.</p>
      </header>
      <ReviewMatrixWorkspace />
    </main>
  );
}
