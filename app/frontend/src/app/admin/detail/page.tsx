import { Suspense } from "react";
import { AdminRecognitionDetail } from "@/features/admin/AdminRecognitionDetail";

export const metadata = { title: "Разбор распознавания" };

export default function AdminDetailPage() {
  return <main className="inner-page inner-page--wide"><Suspense fallback={<div className="detail-loading"><div className="loader" /></div>}><AdminRecognitionDetail /></Suspense></main>;
}
