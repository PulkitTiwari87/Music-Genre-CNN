import "../lab.css";
import { Reproduce } from "../sections/Narrative";

/** The existing "how these numbers were generated" panel, loaded only when the footer is opened. */
export default function ReproducePanel() {
  return (
    <div className="lab">
      <Reproduce />
    </div>
  );
}
