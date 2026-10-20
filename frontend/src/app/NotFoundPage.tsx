import { ButtonLink } from "../ui/Button";
import { EmptyState } from "../ui/EmptyState";
import { icons } from "../ui/icons";

export function NotFoundPage() {
  return (
    <EmptyState
      layout="page"
      icon={icons.attention}
      title="Page not found"
      action={
        <ButtonLink to="/videos" icon={icons.back}>
          Go to videos
        </ButtonLink>
      }
    />
  );
}
