import { AppLayout } from "@/components/layout/app-layout";

export default function EmployeeLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <AppLayout>{children}</AppLayout>;
}
