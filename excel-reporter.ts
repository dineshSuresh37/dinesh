import {
  Reporter,
  TestCase,
  TestResult,
} from '@playwright/test/reporter';
import ExcelJS from 'exceljs';
import path from 'path';

interface TestRow {
  name: string;
  status: string;
}

class ExcelReporter implements Reporter {
  private results: TestRow[] = [];

  onTestEnd(test: TestCase, result: TestResult): void {
    this.results.push({
      name: test.title,
      status: result.status.charAt(0).toUpperCase() + result.status.slice(1),
    });
  }

  async onEnd(): Promise<void> {
    const workbook = new ExcelJS.Workbook();
    const sheet = workbook.addWorksheet('Test Results');

    sheet.columns = [
      { header: 'Test Case Name', key: 'name', width: 40 },
      { header: 'Status', key: 'status', width: 15 },
    ];

    // Style header row
    const headerRow = sheet.getRow(1);
    headerRow.font = { bold: true };
    headerRow.fill = {
      type: 'pattern',
      pattern: 'solid',
      fgColor: { argb: 'FF4472C4' },
    };
    headerRow.font = { bold: true, color: { argb: 'FFFFFFFF' } };

    // Add data rows with status-based color coding
    for (const result of this.results) {
      const row = sheet.addRow(result);
      const statusCell = row.getCell('status');
      if (result.status === 'Passed') {
        statusCell.fill = {
          type: 'pattern',
          pattern: 'solid',
          fgColor: { argb: 'FFC6EFCE' },
        };
        statusCell.font = { color: { argb: 'FF276221' } };
      } else if (result.status === 'Failed') {
        statusCell.fill = {
          type: 'pattern',
          pattern: 'solid',
          fgColor: { argb: 'FFFFC7CE' },
        };
        statusCell.font = { color: { argb: 'FF9C0006' } };
      }
    }

    const outputPath = path.resolve('test-results.xlsx');
    await workbook.xlsx.writeFile(outputPath);
    console.log(`\nExcel report saved to: ${outputPath}`);
  }
}

export default ExcelReporter;
