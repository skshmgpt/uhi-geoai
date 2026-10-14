import { createApp } from './app.js';
import { createDb } from './db.js';

const port = Number(process.env.PORT ?? 8080);
const year = Number(process.env.UHI_YEAR ?? 2024);
const db = createDb(process.env.DATABASE_URL ?? 'postgres://uhi:uhi@localhost:5434/uhi');

createApp({ db, year }).listen(port, () => console.log(`uhi api on :${port}, year ${year}`));
