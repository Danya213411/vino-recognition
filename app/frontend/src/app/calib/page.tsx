import { CalibrationWorkspace } from "@/features/calibration/CalibrationWorkspace";

export const metadata = { title: "Калибровка" };

export default function CalibrationPage() {
  return (
    <main className="inner-page">
      <header className="page-intro">
        <span className="eyebrow">Полевое тестирование</span>
        <h1>Калибровочная<br />лаборатория</h1>
        <p>Проверьте распознавание прямо у полки, укажите правильный ответ и скачайте воспроизводимый манифест прогона.</p>
      </header>
      <CalibrationWorkspace />
    </main>
  );
}
