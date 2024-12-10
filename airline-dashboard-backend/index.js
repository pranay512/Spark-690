// index.js
import dotenv from 'dotenv';
import express, { json } from 'express';
import cors from 'cors';
import { connect, Schema, model } from 'mongoose';
import { Parser } from 'json2csv';

dotenv.config();
const app = express();
const PORT = process.env.PORT || 3001;

// Define CORS options
const corsOptions = {
  origin: 'https://airline-dashboard-chi.vercel.app', // Replace with your actual frontend URL
  optionsSuccessStatus: 200,
};

app.use(cors(corsOptions));
app.use(json());

const MONGODB_URI = process.env.MONGODB_URI;

connect(MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => {
  console.log('Connected to MongoDB');
})
.catch((err) => {
  console.error('MongoDB connection error:', err);
});

// Utility function to parse numbers from nested objects
function parseNumber(v) {
  if (v && v['$numberInt']) {
    return parseInt(v['$numberInt'], 10);
  } else if (v && v['$numberLong']) {
    return parseInt(v['$numberLong'], 10);
  } else if (v && v['$numberDouble']) {
    return parseFloat(v['$numberDouble']);
  }
  return v;
}

// Define Mongoose schema with all variables
const airlineDataSchema = new Schema({
  YEAR: { type: Number, get: parseNumber },
  QUARTER: { type: Number, get: parseNumber },
  UNIQUE_CARRIER: String,
  AIRLINE_ID: { type: Number, get: parseNumber },
  UNIQUE_CARRIER_NAME: String,
  UNIQUE_CARRIER_ENTITY: String,
  REGION: String,
  CARRIER: String,
  CARRIER_NAME: String,
  CARRIER_GROUP_NEW: { type: Number, get: parseNumber },
  DEPARTURES_SCHEDULED: { type: Number, get: parseNumber },
  DEPARTURES_PERFORMED: { type: Number, get: parseNumber },
  PAYLOAD: { type: Number, get: parseNumber },
  SEATS: { type: Number, get: parseNumber },
  PASSENGERS: { type: Number, get: parseNumber },
  FREIGHT: { type: Number, get: parseNumber },
  MAIL: { type: Number, get: parseNumber },
  DISTANCE: { type: Number, get: parseNumber },
  RAMP_TO_RAMP: { type: Number, get: parseNumber },
  AIR_TIME: { type: Number, get: parseNumber },
  NET_INCOME: { type: Number, get: parseNumber },
  OP_PROFIT_LOSS: { type: Number, get: parseNumber },
  TRANS_REV_PAX: { type: Number, get: parseNumber },
  MAIL_REV: { type: Number, get: parseNumber },
  TOTAL_PROPERTY: { type: Number, get: parseNumber },
  PROP_FREIGHT: { type: Number, get: parseNumber },
  PROP_BAG: { type: Number, get: parseNumber },
  TOTAL_CHARTER: { type: Number, get: parseNumber },
  CHARTER_PAX: { type: Number, get: parseNumber },
  MISC_OP_REV: { type: Number, get: parseNumber },
  TRANS_REVENUE: { type: Number, get: parseNumber },
  OP_REVENUES: { type: Number, get: parseNumber },
  FLYING_OPS: { type: Number, get: parseNumber },
  MAINTENANCE: { type: Number, get: parseNumber },
  PAX_SERVICE: { type: Number, get: parseNumber },
  AIRCFT_SERVICES: { type: Number, get: parseNumber },
  PROMOTION_SALES: { type: Number, get: parseNumber },
  GENERAL_ADMIN: { type: Number, get: parseNumber },
  DEPREC_AMORT: { type: Number, get: parseNumber },
  TRANS_EXPENSES: { type: Number, get: parseNumber },
  OP_EXPENSES: { type: Number, get: parseNumber },
  INTEREST_LONG_DEBT: { type: Number, get: parseNumber },
  INTEREST_EXP_OTH: { type: Number, get: parseNumber },
  CAP_GAINS_PROP: { type: Number, get: parseNumber },
  OTHER_INCOME_NET: { type: Number, get: parseNumber },
  NON_OP_INCOME: { type: Number, get: parseNumber },
  INCOME_PRE_TAX: { type: Number, get: parseNumber },
  INCOME_TAX: { type: Number, get: parseNumber },
  INCOME_BEFORE_OTH: { type: Number, get: parseNumber },
  CARRIER_GROUP: { type: Number, get: parseNumber },
  ASM: { type: Number, get: parseNumber },
  RPM: { type: Number, get: parseNumber },
  LOAD_FACTOR: { type: Number, get: parseNumber },
  YIELD: { type: Number, get: parseNumber },
  RASM: { type: Number, get: parseNumber },
  CASM: { type: Number, get: parseNumber },
  PRASM: { type: Number, get: parseNumber },
}, {
  toJSON: { getters: true },
  toObject: { getters: true },
});

const AirlineData = model('AirlineData', airlineDataSchema, 'data'); // 'data' is the collection name

// Define Mongoose schema for stock data
const stockDataSchema = new Schema({
  UNIQUE_CARRIER_NAME: String,
  Date: { type: Date },
  Open: { type: Number, get: parseNumber },
  High: { type: Number, get: parseNumber },
  Low: { type: Number, get: parseNumber },
  Close: { type: Number, get: parseNumber },
  'Adj Close': { type: Number, get: parseNumber },
  Volume: { type: Number, get: parseNumber }
}, {
  toJSON: { getters: true },
  toObject: { getters: true },
});

const StockData = model('StockData', stockDataSchema, 'stock_data'); // 'stock_data' is the collection name

const operatingDataSchema = new Schema({
  PILOT_FLY_OPS: { type: Number, get: parseNumber },
  OTH_FLT_FLY_OPS: { type: Number, get: parseNumber },
  TRAIN_FLY_OPS: { type: Number, get: parseNumber },
  PERS_EXP_FLY_OPS: { type: Number, get: parseNumber },
  PRO_FLY_OPS: { type: Number, get: parseNumber },
  INTERCHG_FLY_OPS: { type: Number, get: parseNumber },
  FUEL_FLY_OPS: { type: Number, get: parseNumber },
  OIL_FLY_OPS: { type: Number, get: parseNumber },
  RENTAL_FLY_OPS: { type: Number, get: parseNumber },
  OTHER_FLY_OPS: { type: Number, get: parseNumber },
  INS_FLY_OPS: { type: Number, get: parseNumber },
  BENEFITS_FLY_OPS: { type: Number, get: parseNumber },
  INCIDENT_FLY_OPS: { type: Number, get: parseNumber },
  PAY_TAX_FLY_OPS: { type: Number, get: parseNumber },
  OTH_TAX_FLY_OPS: { type: Number, get: parseNumber },
  OTHER_EXP_FLY_OPS: { type: Number, get: parseNumber },
  TOT_FLY_OPS: { type: Number, get: parseNumber },
  AIRFRAME_LABOR: { type: Number, get: parseNumber },
  ENGINE_LABOR: { type: Number, get: parseNumber },
  AIRFRAME_REPAIR: { type: Number, get: parseNumber },
  ENGINE_REPAIRS: { type: Number, get: parseNumber },
  INTERCHG_CHARG: { type: Number, get: parseNumber },
  AIRFRAME_MATERIALS: { type: Number, get: parseNumber },
  ENGINE_MATERIALS: { type: Number, get: parseNumber },
  AIRFRAME_ALLOW: { type: Number, get: parseNumber },
  AIRFRAME_OVERHAULS: { type: Number, get: parseNumber },
  ENGINE_ALLOW: { type: Number, get: parseNumber },
  ENGINE_OVERHAULS: { type: Number, get: parseNumber },
  TOT_DIR_MAINT: { type: Number, get: parseNumber },
  AP_MT_BURDEN: { type: Number, get: parseNumber },
  TOT_FLT_MAINT_MEMO: { type: Number, get: parseNumber },
  NET_OBSOL_PARTS: { type: Number, get: parseNumber },
  AIRFRAME_DEP: { type: Number, get: parseNumber },
  ENGINE_DEP: { type: Number, get: parseNumber },
  PARTS_DEP: { type: Number, get: parseNumber },
  ENG_PARTS_DEP: { type: Number, get: parseNumber },
  OTH_FLT_EQUIP_DEP: { type: Number, get: parseNumber },
  OTH_FLT_EQUIP_DEP_GRP_I: { type: Number, get: parseNumber },
  FLT_EQUIP_A_EXP: { type: Number, get: parseNumber },
  FLY_OPS_EXP_I_A: { type: Number, get: parseNumber },
  TOT_AIR_OP_EXPENSES: { type: Number, get: parseNumber },
  DEV_N_PREOP_EXP: { type: Number, get: parseNumber },
  OTH_INTANGIBLES: { type: Number, get: parseNumber },
  EQUIP_N_HANGAR_DEP: { type: Number, get: parseNumber },
  G_PROP_DEP: { type: Number, get: parseNumber },
  CAP_LEASES_DEP: { type: Number, get: parseNumber },
  TOTAL_AIR_HOURS: { type: Number, get: parseNumber },
  AIR_DAYS_ASSIGN: { type: Number, get: parseNumber },
  AIR_FUELS_ISSUED: { type: Number, get: parseNumber },
  AIRCRAFT_CONFIG: String,
  AIRCRAFT_GROUP: String,
  AIRCRAFT_TYPE: String,
  AIRLINE_ID: { type: Number, get: parseNumber },
  UNIQUE_CARRIER_NAME: String,
  REGION: String,
  CARRIER_GROUP_NEW: { type: Number, get: parseNumber },
  YEAR: { type: Number, get: parseNumber },
  QUARTER: { type: Number, get: parseNumber },
  SSD_NAME: String,
  MANUFACTURER: String,
  SHORT_NAME: String,
  AIRCRAFT_CATEGORIZATION: String,
  AVL_SEAT_MILES_320: { type: Number, get: parseNumber },
  REV_PAX_MILES_140: { type: Number, get: parseNumber },
  REV_TON_MILES_240: { type: Number, get: parseNumber },
  REV_TON_MILES_FREIGHT_247: { type: Number, get: parseNumber },
  REV_TON_MILES_MAIL_249: { type: Number, get: parseNumber },
  AVL_TON_MILES_280: { type: Number, get: parseNumber },
  REV_ACRFT_MILES_FLOWN_410: { type: Number, get: parseNumber },
  REV_ACRFT_DEP_PERF_510: { type: Number, get: parseNumber },
  REV_ACRFT_HRS_AIRBORNE_610: { type: Number, get: parseNumber },
  NON_REV_ACRFT_HRS_AIRBORNE_620: { type: Number, get: parseNumber },
  ACRFT_HRS_RAMPTORAMP_630: { type: Number, get: parseNumber },
  HOURS_AIRBORNE_650: { type: Number, get: parseNumber },
  AIR_DAYS_EQUIP_810: { type: Number, get: parseNumber },
  AIR_DAYS_ROUTE_820: { type: Number, get: parseNumber },
  AIRCRAFT_FUELS_921: { type: Number, get: parseNumber },
}, {
  toJSON: { getters: true },
  toObject: { getters: true },
});

// Create the Mongoose model
const OperatingData = model('OperatingData', operatingDataSchema, 'operating');

const operatingDataExtendedSchema = new Schema({
  // Salary Columns
  SALARIES_MGT: { type: Number, get: parseNumber },
  SALARIES_FLIGHT: { type: Number, get: parseNumber },
  SALARIES_MAINT: { type: Number, get: parseNumber }, 
  SALARIES_TRAFFIC: { type: Number, get: parseNumber },
  SALARIES_OTHER: { type: Number, get: parseNumber },
  SALARIES: { type: Number, get: parseNumber },

  // Benefits Columns
  BENEFITS_PERSONNEL: { type: Number, get: parseNumber },
  BENEFITS_PENSIONS: { type: Number, get: parseNumber },
  BENEFITS_PAYROLL: { type: Number, get: parseNumber },
  BENEFITS: { type: Number, get: parseNumber },
  SALARIES_BENEFITS: { type: Number, get: parseNumber },

  // Materials Columns  
  AIRCRAFT_FUEL: { type: Number, get: parseNumber },
  MAINT_MATERIAL: { type: Number, get: parseNumber },
  FOOD: { type: Number, get: parseNumber },
  OTHER_MATERIALS: { type: Number, get: parseNumber },
  MATERIALS_TOTAL: { type: Number, get: parseNumber },

  // Services Columns
  ADVERTISING: { type: Number, get: parseNumber },
  COMMUNICATION: { type: Number, get: parseNumber },
  INSURANCE: { type: Number, get: parseNumber },
  OUTSIDE_EQUIP: { type: Number, get: parseNumber },
  COMMISIONS_PAX: { type: Number, get: parseNumber },
  COMMISSIONS_CARGO: { type: Number, get: parseNumber },
  OTHER_SERVICES: { type: Number, get: parseNumber },
  SERVICES_TOTAL: { type: Number, get: parseNumber },

  // Other Operational Expenses
  LANDING_FEES: { type: Number, get: parseNumber },
  RENTALS: { type: Number, get: parseNumber },
  DEPRECIATION: { type: Number, get: parseNumber },
  AMORTIZATION: { type: Number, get: parseNumber },
  OTHER: { type: Number, get: parseNumber },
  TRANS_EXPENSE: { type: Number, get: parseNumber },
  OP_EXPENSE: { type: Number, get: parseNumber },

  // Traffic Metrics
  REV_PAX_ENP_110: { type: Number, get: parseNumber },
  REV_PAX_MILES_140: { type: Number, get: parseNumber },
  REV_PAX_MILES_COACH: { type: Number, get: parseNumber },
  REV_PAX_MILES_FIRST: { type: Number, get: parseNumber },
  NON_REV_PAX_MILES: { type: Number, get: parseNumber },
  REV_TON_MILES_240: { type: Number, get: parseNumber },
  REV_TON_MILES_FREIGHT_247: { type: Number, get: parseNumber },
  REV_TON_MILES_PAX_241: { type: Number, get: parseNumber },
  REV_TON_MILES_USPR_MAIL: { type: Number, get: parseNumber },
  REV_TON_MILES_USNONPR_MAIL: { type: Number, get: parseNumber },
  REV_TON_MILES_MAIL_249: { type: Number, get: parseNumber },
  REV_TON_MILES_FOREIGN_MAIL: { type: Number, get: parseNumber },
  REV_TON_MILES_EXPRESS_MAIL: { type: Number, get: parseNumber },
  NON_REV_TON_MILES: { type: Number, get: parseNumber },
  AVL_TON_MILES_280: { type: Number, get: parseNumber },
  AVL_SEAT_MILES_320: { type: Number, get: parseNumber },
  AVL_SEAT_MILES_FIRST: { type: Number, get: parseNumber },
  AVL_SEAT_MILES_COACH: { type: Number, get: parseNumber },
  REV_ACRFT_MILES_FLOWN_410: { type: Number, get: parseNumber },
  REV_ACRFT_MILES_SCH_430: { type: Number, get: parseNumber },
  REV_ACRFT_MILES_SCH_COMPLETED: { type: Number, get: parseNumber },
  REV_ACRFT_DEP_PERF_510: { type: Number, get: parseNumber },
  REV_ACRFT_HRS_AIRBORNE_610: { type: Number, get: parseNumber },
  ACRFT_HRS_RAMPTORAMP_630: { type: Number, get: parseNumber },

  // Carrier and Identification Columns
  AIRLINE_ID: { type: Number, get: parseNumber },
  UNIQUE_CARRIER: String,
  UNIQUE_CARRIER_NAME: String, 
  REGION: String,

  // Temporal Columns
  YEAR: { type: Number, get: parseNumber },
  QUARTER: { type: Number, get: parseNumber }
}, {
  toJSON: { getters: true },
  toObject: { getters: true },
});

// Create the Mongoose model
const OperatingDataExtended = model('OperatingDataExtended', operatingDataExtendedSchema, 'operatingdataextended');

// New Schema for Balance Sheets
const balanceSheetsSchema = new Schema({
  AIRLINE_ID: { type: Number, get: parseNumber },
  UNIQUE_CARRIER_NAME: String,
  YEAR: { type: Number, get: parseNumber },
  QUARTER: { type: Number, get: parseNumber },
  CASH: { type: Number, get: parseNumber },
  SHORT_TERM_INV: { type: Number, get: parseNumber },
  NOTES_RECEIVABLE: { type: Number, get: parseNumber },
  ACCTS_RECEIVABLE: { type: Number, get: parseNumber },
  ACCTS_NOT_COLLECT: { type: Number, get: parseNumber },
  NOTES_ACC_REC_NET: { type: Number, get: parseNumber },
  PARTS_SUPPLIES_NET: { type: Number, get: parseNumber },
  PREPAID_ITEMS: { type: Number, get: parseNumber },
  CURR_ASSETS_OTH: { type: Number, get: parseNumber },
  CURR_ASSETS: { type: Number, get: parseNumber },
  INVEST_ASSOC_COMP: { type: Number, get: parseNumber },
  INVEST_REC_OTH: { type: Number, get: parseNumber },
  SPECIAL_FUNDS: { type: Number, get: parseNumber },
  INVEST_SPEC_FUNDS: { type: Number, get: parseNumber },
  FLIGHT_EQUIP: { type: Number, get: parseNumber },
  PROP_EQUIP_GROUND: { type: Number, get: parseNumber },
  DEPR_PR_EQ_GROUND: { type: Number, get: parseNumber },
  PROP_EQUIP_NET: { type: Number, get: parseNumber },
  LAND: { type: Number, get: parseNumber },
  EQUIP_DEP_ADV_PAY: { type: Number, get: parseNumber },
  CONSTRUCTION: { type: Number, get: parseNumber },
  LEASED_PROP_CAP: { type: Number, get: parseNumber },
  LEASED_PROP_ACC: { type: Number, get: parseNumber },
  PROP_EQUIP: { type: Number, get: parseNumber },
  PROP_EQUIP_NON_OP: { type: Number, get: parseNumber },
  DEPR_PR_EQ_NON_OP: { type: Number, get: parseNumber },
  PROP_EQUIP_NO_TOT: { type: Number, get: parseNumber },
  PRE_PAY_LONG_TERM: { type: Number, get: parseNumber },
  NON_AMORT_DEV: { type: Number, get: parseNumber },
  ASSETS_OTH_DEF: { type: Number, get: parseNumber },
  ASSETS_OTHER: { type: Number, get: parseNumber },
  ASSETS: { type: Number, get: parseNumber },
  LONG_DEBT_CUR_MAT: { type: Number, get: parseNumber },
  NOTES_PAY_BANKS: { type: Number, get: parseNumber },
  NOTES_PAY_OTHER: { type: Number, get: parseNumber },
  ACCTS_PAY_TRADE: { type: Number, get: parseNumber },
  ACCTS_PAY_OTHER: { type: Number, get: parseNumber },
  CURR_OB_CAP_LEASE: { type: Number, get: parseNumber },
  ACCR_SALARIES: { type: Number, get: parseNumber },
  ACCR_VACATION: { type: Number, get: parseNumber },
  ACCR_INTEREST: { type: Number, get: parseNumber },
  ACCR_TAXES: { type: Number, get: parseNumber },
  DIVIDENDS: { type: Number, get: parseNumber },
  LIAB_AIR_TRAFFIC: { type: Number, get: parseNumber },
  CURR_LIAB_OTH: { type: Number, get: parseNumber },
  CURR_LIABILITIES: { type: Number, get: parseNumber },
  LONG_TERM_DEBT: { type: Number, get: parseNumber },
  ADV_ASSOC_COMP: { type: Number, get: parseNumber },
  PENSION_LIAB: { type: Number, get: parseNumber },
  NON_REC_OB_CAP_LS: { type: Number, get: parseNumber },
  NON_REC_LIAB_OTH: { type: Number, get: parseNumber },
  NON_REC_LIAB: { type: Number, get: parseNumber },
  DEF_TAXES: { type: Number, get: parseNumber },
  DEF_TAX_CREDITS: { type: Number, get: parseNumber },
  DEF_CREDITS_OTH: { type: Number, get: parseNumber },
  DEF_CREDITS: { type: Number, get: parseNumber },
  PF_SHARES: { type: Number, get: parseNumber },
  PF_SHARES_NUM: { type: Number, get: parseNumber },
  COM_SHARES: { type: Number, get: parseNumber },
  UNISSUED_STOCK: { type: Number, get: parseNumber },
  CAPITAL_STOCK: { type: Number, get: parseNumber },
  ADD_CAPITAL_INV: { type: Number, get: parseNumber },
  PAID_IN_CAPITAL: { type: Number, get: parseNumber },
  RET_EARNINGS: { type: Number, get: parseNumber },
  SH_HLD_EQUITY: { type: Number, get: parseNumber },
  TREAS_STOCK_NUM: { type: Number, get: parseNumber },
  SH_HLD_EQUIT_NET: { type: Number, get: parseNumber },
  LIAB_SH_HLD_EQUITY: { type: Number, get: parseNumber },
}, {
  toJSON: { getters: true },
  toObject: { getters: true },
});

const BalanceSheets = model('BalanceSheets', balanceSheetsSchema, 'balanceSheets');

// List of airlines with IDs matching the frontend
const airlines = [
  {
    id: 'alaska-airlines',
    name: 'Alaska Airlines Inc.',
    nasdaqName: 'Alaska Air Group Inc.',
    ticker: 'ALK',
    category: 'legacy',
  },
  {
    id: 'allegiant-air',
    name: 'Allegiant Air',
    nasdaqName: 'Allegiant Travel Company',
    ticker: 'ALGT',
    category: 'ultra low-cost',
  },
  {
    id: 'american-airlines',
    name: 'American Airlines Inc.',
    nasdaqName: 'American Airlines Group Inc.',
    ticker: 'AAL',
    category: 'legacy',
  },
  {
    id: 'delta-airlines',
    name: 'Delta Air Lines Inc.',
    nasdaqName: 'Delta Air Lines Inc.',
    ticker: 'DAL',
    category: 'legacy',
  },
  {
    id: 'frontier-airlines',
    name: 'Frontier Airlines Inc.',
    nasdaqName: 'Frontier Group Holdings, Inc.',
    ticker: 'ULCC',
    category: 'ultra low-cost',
  },
  {
    id: 'jetblue-airlines',
    name: 'JetBlue Airways',
    nasdaqName: 'JetBlue Airways Corporation',
    ticker: 'JBLU',
    category: 'low-cost',
  },
  {
    id: 'hawaiian-airlines',
    name: 'Hawaiian Airlines Inc.',
    nasdaqName: 'Hawaiian Holdings Inc.',
    ticker: 'HA',
    category: 'legacy',
  },
  {
    id: 'southwest-airlines',
    name: 'Southwest Airlines Co.',
    nasdaqName: 'Southwest Airlines Co.',
    ticker: 'LUV',
    category: 'low-cost',
  },
  {
    id: 'spirit-airlines',
    name: 'Spirit Air Lines',
    nasdaqName: 'Spirit Airlines, Inc.',
    ticker: 'SAVE',
    category: 'ultra low-cost',
  },
  {
    id: 'united-airlines',
    name: 'United Air Lines Inc.',
    nasdaqName: 'United Airlines Holdings, Inc.',
    ticker: 'UAL',
    category: 'legacy',
  },
];

// Endpoint to get airline data
app.get('/api/airline-data/:airlineId', async (req, res) => {
  const { airlineId } = req.params;

  const airline = airlines.find((a) => a.id === airlineId);

  if (!airline) {
    return res.status(404).json({ error: 'Airline not found' });
  }

  try {
    // Query MongoDB for the airline data
    const airlineData = await AirlineData.find({
      UNIQUE_CARRIER_NAME: airline.name,
    }).exec();

    console.log(`Data for ${airline.name}:`, airlineData.length); // Debugging line

    res.json(airlineData);
  } catch (err) {
    console.error('Error querying MongoDB:', err);
    res.status(500).json({ error: 'Failed to retrieve data from MongoDB' });
  }
});

// Endpoint to get stock data
app.get('/api/stock-data/:airlineId', async (req, res) => {
  const { airlineId } = req.params;

  console.log('Received request for airlineId:', airlineId);

  const airline = airlines.find((a) => a.id === airlineId);

  console.log('Matched airline:', airline);

  if (!airline) {
    return res.status(404).json({ error: 'Airline not found' });
  }

  try {
    // Query MongoDB for the stock data
    const stockData = await StockData.find({
      UNIQUE_CARRIER_NAME: airline.name,
    }).exec();

    console.log(`Stock data for ${airline.name}:`, stockData.length); // Debugging line

    res.json(stockData);
  } catch (err) {
    console.error('Error querying MongoDB:', err);
    res.status(500).json({ error: 'Failed to retrieve stock data from MongoDB' });
  }
});

app.get('/api/stock-kpis', async (req, res) => {
  try {
    // Fetch stock data for all airlines
    const stockData = await StockData.find({}).exec();

    // Map airlines by name for easy lookup
    const airlineMap = airlines.reduce((acc, airline) => {
      acc[airline.name] = airline;
      return acc;
    }, {});

    // Organize stock data by airline
    const stockDataByAirline = {};
    stockData.forEach(item => {
      const airlineName = item.UNIQUE_CARRIER_NAME;
      if (!stockDataByAirline[airlineName]) {
        stockDataByAirline[airlineName] = [];
      }
      stockDataByAirline[airlineName].push(item);
    });

    // Compute KPIs for each airline
    const allStockKPIs = Object.keys(stockDataByAirline).map(airlineName => {
      const data = stockDataByAirline[airlineName];

      const sortedData = data
        .map(item => ({
          ...item.toObject(),
          Date: new Date(item.Date)
        }))
        .sort((a, b) => a.Date - b.Date);

      // Get latest data point
      const latestDataPoint = sortedData[sortedData.length - 1];
      const latestPrice = parseFloat(latestDataPoint['Adj Close']);
      const latestDate = latestDataPoint.Date;

      // 1-Year Return calculation
      const oneYearAgoDate = new Date(latestDate);
      oneYearAgoDate.setFullYear(oneYearAgoDate.getFullYear() - 1);
      const oneYearAgoDataPoint = sortedData.find(d => d.Date >= oneYearAgoDate) || 
                                  sortedData.find(d => d.Date <= oneYearAgoDate);
      let oneYearReturn;
      if (oneYearAgoDataPoint && oneYearAgoDataPoint['Adj Close']) {
        const startPrice = parseFloat(oneYearAgoDataPoint['Adj Close']);
        oneYearReturn = ((latestPrice - startPrice) / startPrice) * 100;
      } else {
        oneYearReturn = 'N/A';
      }

      // 3-Year Return calculation
      const threeYearsAgoDate = new Date(latestDate);
      threeYearsAgoDate.setFullYear(threeYearsAgoDate.getFullYear() - 3);
      const threeYearsAgoDataPoint = sortedData.find(d => d.Date >= threeYearsAgoDate) || 
                                     sortedData.find(d => d.Date <= threeYearsAgoDate);
      let threeYearReturn;
      if (threeYearsAgoDataPoint && threeYearsAgoDataPoint['Adj Close']) {
        const startPrice = parseFloat(threeYearsAgoDataPoint['Adj Close']);
        threeYearReturn = ((latestPrice - startPrice) / startPrice) * 100;
      } else {
        threeYearReturn = 'N/A';
      }

      // 5-Year Return calculation
      const fiveYearsAgoDate = new Date(latestDate);
      fiveYearsAgoDate.setFullYear(fiveYearsAgoDate.getFullYear() - 5);
      const fiveYearsAgoDataPoint = sortedData.find(d => d.Date >= fiveYearsAgoDate) || 
                                    sortedData.find(d => d.Date <= fiveYearsAgoDate);
      let fiveYearReturn;
      if (fiveYearsAgoDataPoint && fiveYearsAgoDataPoint['Adj Close']) {
        const startPrice = parseFloat(fiveYearsAgoDataPoint['Adj Close']);
        fiveYearReturn = ((latestPrice - startPrice) / startPrice) * 100;
      } else {
        fiveYearReturn = 'N/A';
      }

      // Get airline info
      const airline = airlineMap[airlineName];
      if (!airline) {
        return null;
      }

      return {
        nasdaqName: airline.nasdaqName,
        ticker: airline.ticker,
        airlineId: airline.id,
        latestPrice,
        latestDate: latestDate.toLocaleDateString(),
        oneYearReturn,
        threeYearReturn,
        fiveYearReturn,
      };
    }).filter(kpi => kpi !== null);

    res.json(allStockKPIs);
  } catch (err) {
    console.error('Error fetching all stock KPIs:', err);
    res.status(500).json({ error: 'Failed to fetch stock KPIs' });
  }
});

// Endpoint to get operating data
app.get('/api/operating-data/:airlineId', async (req, res) => {
  const { airlineId } = req.params;

  const airline = airlines.find((a) => a.id === airlineId);

  if (!airline) {
    return res.status(404).json({ error: 'Airline not found' });
  }

  try {
    // Query MongoDB for the operating data
    const operatingData = await OperatingData.find({
      UNIQUE_CARRIER_NAME: airline.name,
    }).exec();

    console.log(`Operating data for ${airline.name}:`, operatingData.length); // Debugging line

    res.json(operatingData);
  } catch (err) {
    console.error('Error querying MongoDB:', err);
    res.status(500).json({ error: 'Failed to retrieve operating data from MongoDB' });
  }
});

// Endpoint to get extended operating data
app.get('/api/operatingdataextended/:airlineId', async (req, res) => {
  const { airlineId } = req.params;

  const airline = airlines.find((a) => a.id === airlineId);

  if (!airline) {
    return res.status(404).json({ error: 'Airline not found' });
  }

  try {
    // Query MongoDB for the operating data extended
    const operatingDataExtended = await OperatingDataExtended.find({
      UNIQUE_CARRIER_NAME: airline.name,
    }).exec();

    console.log(`Operating data Extended for ${airline.name}:`, operatingDataExtended.length); // Debugging line

    res.json(operatingDataExtended);
  } catch (err) {
    console.error('Error querying MongoDB:', err);
    res.status(500).json({ error: 'Failed to retrieve operating data from MongoDB' });
  }
});

// Endpoint to get balance sheet data
app.get('/api/balance-sheets/:airlineId', async (req, res) => {
  const { airlineId } = req.params;

  const airline = airlines.find((a) => a.id === airlineId);

  if (!airline) {
    return res.status(404).json({ error: 'Airline not found' });
  }

  try {
    // Query MongoDB for the balance sheet data
    const balanceSheetData = await BalanceSheets.find({
      UNIQUE_CARRIER_NAME: airline.name,
    }).exec();

    console.log(`Balance sheet data for ${airline.name}:`, balanceSheetData.length); // Debugging line

    res.json(balanceSheetData);
  } catch (err) {
    console.error('Error querying MongoDB:', err);
    res.status(500).json({ error: 'Failed to retrieve balance sheet data from MongoDB' });
  }
});

// Download Endpoint (Dynamic CSV Generation)
app.get('/api/download/:collectionName', async (req, res) => {
  const { collectionName } = req.params;

  // Map collection names to Mongoose models
  const modelMap = {
    'airline_data': AirlineData,
    'operating': OperatingData,
    'operatingdataextended': OperatingDataExtended,
    'stock_data': StockData,
    'balanceSheets': BalanceSheets, // Added new collection here
  };

  const Model = modelMap[collectionName];

  if (!Model) {
    return res.status(404).send('File not found');
  }

  try {
    // Fetch data from MongoDB
    const data = await Model.find({}).lean().exec();

    if (!data || data.length === 0) {
      return res.status(404).send('No data available for this collection');
    }

    // Convert JSON to CSV
    const json2csvParser = new Parser();
    const csv = json2csvParser.parse(data);

    // Set headers for CSV download
    res.header('Content-Type', 'text/csv');
    res.attachment(`${collectionName}.csv`);
    return res.send(csv);
  } catch (err) {
    console.error(`Error generating CSV for ${collectionName}:`, err);
    return res.status(500).send('Failed to generate CSV file');
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
