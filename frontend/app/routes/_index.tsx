import {MetaFunction, redirect} from "@remix-run/node";

export const meta: MetaFunction = () => {
  return [
    { title: "New Remix App" },
    { name: "description", content: "Welcome to Remix!" },
  ];
};

export async function loader() {
  return redirect("/dashboard");
}

// export default function Index() {
//   return DashBoard();
// }