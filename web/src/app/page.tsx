import { Dashboard } from "@/components/dashboard/dashboard";
import { getMeta, getPredictions, getPriceSentiment } from "@/lib/data";

export default async function Page() {
  const [meta, prices, predictions] = await Promise.all([getMeta(), getPriceSentiment(), getPredictions()]);
  return <Dashboard meta={meta} prices={prices} predictions={predictions} />;
}
