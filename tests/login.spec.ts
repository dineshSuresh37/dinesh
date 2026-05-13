import { test, expect } from '@playwright/test';

const LOGIN_URL = 'https://practice.expandtesting.com/login';
const VALID_USER = 'practice';
const VALID_PASS = 'SuperSecretPassword!';

test.describe('Login Functionality', () => {
  test('login with valid credentials', async ({ page }) => {
    await page.goto(LOGIN_URL);

    await page.getByLabel('Username').fill(VALID_USER);
    await page.getByLabel('Password').fill(VALID_PASS);
    await page.getByRole('button', { name: 'Login' }).click();

    await expect(page).toHaveURL(/secure/);
    await expect(page.getByText('You logged into a secure area!')).toBeVisible();
  });

  test('login with invalid username', async ({ page }) => {
    await page.goto(LOGIN_URL);

    await page.getByLabel('Username').fill('wronguser');
    await page.getByLabel('Password').fill(VALID_PASS);
    await page.getByRole('button', { name: 'Login' }).click();

    await expect(page).toHaveURL(LOGIN_URL);
    await expect(page.getByText('Your username is invalid!')).toBeVisible();
  });

  test('login with invalid password', async ({ page }) => {
    await page.goto(LOGIN_URL);

    await page.getByLabel('Username').fill(VALID_USER);
    await page.getByLabel('Password').fill('wrongpassword');
    await page.getByRole('button', { name: 'Login' }).click();

    await expect(page).toHaveURL(LOGIN_URL);
    await expect(page.getByText('Your password is invalid!')).toBeVisible();
  });

  test('login with empty credentials', async ({ page }) => {
    await page.goto(LOGIN_URL);

    await page.getByRole('button', { name: 'Login' }).click();

    await expect(page).toHaveURL(LOGIN_URL);
    await expect(page.getByText('Your username is invalid!')).toBeVisible();
  });

  test('logout after successful login', async ({ page }) => {
    await page.goto(LOGIN_URL);

    await page.getByLabel('Username').fill(VALID_USER);
    await page.getByLabel('Password').fill(VALID_PASS);
    await page.getByRole('button', { name: 'Login' }).click();

    await expect(page).toHaveURL(/secure/);
    await page.getByRole('link', { name: 'Logout' }).click();

    await expect(page).toHaveURL(LOGIN_URL);
    await expect(page.getByText('You logged out of the secure area!')).toBeVisible();
  });

  test('practicetestautomation login with valid credentials', async ({ page }) => {
    await page.goto('https://practicetestautomation.com/practice-test-login/');

    await page.getByLabel('Username').fill('student');
    await page.getByLabel('Password').fill('Password123');
    await page.getByRole('button', { name: 'Submit' }).click();

    await expect(page.getByText('Logged In Successfully')).toBeVisible();
  });
});
