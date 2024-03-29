import "@mantine/core/styles.css";
import '@mantine/dropzone/styles.css';
import '@mantine/notifications/styles.css';
import '@mantine/carousel/styles.css';

import {cssBundleHref} from "@remix-run/css-bundle";
import type {LinksFunction} from "@remix-run/node";
import {
    Links,
    LiveReload,
    Meta,
    Outlet,
    Scripts,
    ScrollRestoration,
} from "@remix-run/react";
import {MantineProvider, ColorSchemeScript} from "@mantine/core";
import {Notifications} from "@mantine/notifications";
import {ModalsProvider} from "@mantine/modals";

export const links: LinksFunction = () => [
    ...(cssBundleHref ? [{rel: "stylesheet", href: cssBundleHref}] : []),
];

export default function App() {
    return (
        <html lang="en">
        <head>
            <meta charSet="utf-8"/>
            <meta name="viewport" content="width=device-width, initial-scale=1"/>
            <Meta/>
            <Links/>
            <ColorSchemeScript/>
        </head>
        <body style={{overflow: 'hidden'}}>
        <MantineProvider>
            <ModalsProvider>
                <Notifications/>
                <Outlet/>
                <ScrollRestoration/>
                <Scripts/>
                <LiveReload/>
            </ModalsProvider>
        </MantineProvider>
        </body>
        </html>
    );
}
