import { AdminDashboard } from "@/features/admin/AdminDashboard";

export const metadata = { title: "Администрирование" };

export default function AdminPage() {
  return (
    <main className="inner-page inner-page--wide">
      <header className="page-intro page-intro--row">
        <div><span className="eyebrow">Операционный журнал</span><h1>Администрирование</h1></div>
        <p>Входящие кадры, псевдонимные пользователи, решения модели, скорость и ответы калибровщиков.</p>
      </header>
      <AdminDashboard />
    </main>
  );
}
